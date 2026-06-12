"""Mixed embedding layer for 6 token types."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np  # used for bb_fingerprints buffer

from routeflow.data.route import TokenType


class RouteEmbedding(nn.Module):
    """Unified embedding for all token types in route sequences.

    Token types and their embeddings:
        START (0), END (1):       Embedding(2, d)
        BRANCH (2), REACT (3):    Embedding(2, d)
        POP (4):                  Embedding(1, d)
        RXN (5):                  Embedding(115, d)     - trainable
        BB (6):                   Linear(2048, d)       - FP frozen, Linear trainable
    """

    def __init__(
        self,
        embed_dim: int,
        num_rxn_templates: int,
        bb_fingerprints: np.ndarray,  # (num_bb, fp_nbits), frozen
        fp_nbits: int = 2048,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_rxn = num_rxn_templates
        self.fp_nbits = fp_nbits

        # Special tokens: START=0, END=1
        self.special_emb = nn.Embedding(2, embed_dim)
        # Action tokens: BRANCH=0, REACT=1
        self.action_emb = nn.Embedding(2, embed_dim)
        # Pop token
        self.pop_emb = nn.Embedding(1, embed_dim)
        # Reaction template embedding (trainable)
        self.rxn_emb = nn.Embedding(num_rxn_templates, embed_dim)
        # BB embedding: frozen FP → trainable MLP (3-layer, wide, with LayerNorm)
        self.bb_proj = nn.Sequential(
            nn.Linear(fp_nbits, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

        # Frozen BB fingerprints buffer. persistent=False keeps it out of
        # state_dict — it's already loaded from bb_fingerprints.npy via the
        # constructor, and saving a 2.2GB float32 tensor every checkpoint was
        # blowing host RAM during torch.save (caused OOM kills mid-save).
        # Old checkpoints that still have the key load fine via strict=False.
        self.register_buffer(
            "bb_fps",
            torch.from_numpy(bb_fingerprints).float(),
            persistent=False,
        )  # (num_bb, fp_nbits)

    def forward(
        self,
        token_types: torch.Tensor,    # (B, L) int
        token_values: torch.Tensor,   # (B, L) int
    ) -> torch.Tensor:
        """Compute embeddings for the full token sequence.

        Returns: (B, L, embed_dim)
        """
        B, L = token_types.shape
        device = token_types.device
        emb = torch.zeros(B, L, self.embed_dim, device=device)

        # START / END
        mask_start = token_types == TokenType.START
        mask_end = token_types == TokenType.END
        if mask_start.any():
            emb[mask_start] = self.special_emb(torch.zeros_like(token_values[mask_start]))
        if mask_end.any():
            emb[mask_end] = self.special_emb(torch.ones_like(token_values[mask_end]))

        # BRANCH / REACT
        mask_branch = token_types == TokenType.BRANCH
        mask_react = token_types == TokenType.REACT
        if mask_branch.any():
            emb[mask_branch] = self.action_emb(torch.zeros_like(token_values[mask_branch]))
        if mask_react.any():
            emb[mask_react] = self.action_emb(torch.ones_like(token_values[mask_react]))

        # POP
        mask_pop = token_types == TokenType.POP
        if mask_pop.any():
            emb[mask_pop] = self.pop_emb(torch.zeros_like(token_values[mask_pop]))

        # RXN
        mask_rxn = token_types == TokenType.RXN
        if mask_rxn.any():
            rxn_ids = token_values[mask_rxn].clamp(0, self.num_rxn - 1)
            emb[mask_rxn] = self.rxn_emb(rxn_ids)

        # BB: look up frozen FP, project through trainable linear
        mask_bb = token_types == TokenType.BB
        if mask_bb.any():
            bb_ids = token_values[mask_bb].clamp(0, self.bb_fps.shape[0] - 1)
            bb_fp = self.bb_fps[bb_ids]  # (N, fp_nbits)
            emb[mask_bb] = self.bb_proj(bb_fp)

        return emb

    def embed_bb_fingerprint(self, fp: torch.Tensor) -> torch.Tensor:
        """Project BB fingerprint(s) through the BB linear layer.

        Args:
            fp: (..., fp_nbits)
        Returns: (..., embed_dim)
        """
        return self.bb_proj(fp)

    def get_bb_repr_matrix(self) -> torch.Tensor:
        """Return BB representation matrix for dot-product classification.

        Returns: (num_bb, embed_dim) - each row is repr(b_i) = Linear(fp(b_i))
        """
        return self.bb_proj(self.bb_fps)  # (num_bb, embed_dim)

"""TransformerDecoder: autoregressive, closed-loop, stack-based.

Three classifier heads:
  - action_head:  Linear → 3 logits (BRANCH / REACT / END)
  - rxn_head:     Linear → num_rxn logits
  - source_head:  MLP → query → dot(BB_repr) + Pop logit
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from routeflow.models.encoder import PositionalEncoding


class SourceHead(nn.Module):
    """MLP that produces a query vector for dot-product BB classification + Pop logit.

    query = MLP(hidden) → dot(BB_repr_matrix) → logits over ~2K BBs
    Plus a separate Pop logit.
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        # Separate Pop logit head
        self.pop_head = nn.Linear(d_model, 1)

    def forward(
        self,
        h: torch.Tensor,                    # (..., d_model)
        bb_repr_matrix: torch.Tensor,        # (num_bb, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (bb_logits, pop_logit).

        bb_logits: (..., num_bb)
        pop_logit: (..., 1)
        """
        query = self.mlp(h)                  # (..., d_model)
        bb_logits = query @ bb_repr_matrix.T  # (..., num_bb)
        pop_logit = self.pop_head(h)         # (..., 1)
        return bb_logits, pop_logit


class RouteDecoder(nn.Module):
    """Transformer decoder for autoregressive route generation.

    During training (teacher forcing):
        Full sequence known → parallel with causal mask.

    During inference (closed-loop):
        Generate token by token, execute reactions via RDKit,
        inject mol_state FPs from real intermediates.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        latent_dim: int = 128,
        num_rxn: int = 115,
        num_layers: int = 6,
        nhead: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_rxn = num_rxn

        # Project z to decoder memory
        self.z_proj = nn.Linear(latent_dim, embed_dim)

        # Positional encoding for decoder input
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max(max_len, 256), dropout=dropout)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # Classifier heads
        # Action: BRANCH(0) / REACT(1) / END(2)
        self.action_head = nn.Linear(embed_dim, 3)
        # Reaction: over num_rxn templates
        self.rxn_head = nn.Linear(embed_dim, num_rxn)
        # Source: BB selection via dot-product + Pop
        self.source_head = SourceHead(embed_dim)

    def forward(
        self,
        tgt_emb: torch.Tensor,         # (B, L, embed_dim) target token embeddings
        z: torch.Tensor,               # (B, latent_dim)
        tgt_padding_mask: torch.Tensor, # (B, L) True where padded
    ) -> torch.Tensor:
        """Run transformer decoder, return hidden states at all positions.

        Returns: (B, L, embed_dim)
        """
        B, L, _ = tgt_emb.shape

        # Prepare memory from z: (B, 1, embed_dim)
        memory = self.z_proj(z).unsqueeze(1)  # (B, 1, embed_dim)

        # Add positional encoding to target embeddings
        tgt = self.pos_enc(tgt_emb)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, dtype=tgt.dtype, device=tgt.device
        )

        # Run decoder
        h = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return h  # (B, L, embed_dim)

    def predict_action(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., embed_dim) → action logits (..., 3)"""
        return self.action_head(h)

    def predict_rxn(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., embed_dim) → rxn logits (..., num_rxn)"""
        return self.rxn_head(h)

    def predict_source(
        self,
        h: torch.Tensor,
        bb_repr_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (..., embed_dim) → (bb_logits (..., num_bb), pop_logit (..., 1))"""
        return self.source_head(h, bb_repr_matrix)

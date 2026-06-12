"""TransformerEncoder: route token sequence → z ∈ R^128."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RouteEncoder(nn.Module):
    """Transformer encoder that maps route token embeddings → z ∈ R^embed_dim.

    Architecture:
        token embeddings (B, L, embed_dim)
        → PositionalEncoding
        → TransformerEncoder (N layers)
        → mean pool over non-padded positions
        → z ∈ R^embed_dim (= latent_dim, no extra projection)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.pos_enc = PositionalEncoding(embed_dim, max_len=max(max_len, 256), dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )

        # Only add projection if latent_dim != embed_dim
        if latent_dim != embed_dim:
            self.fc_proj = nn.Linear(embed_dim, latent_dim)
        else:
            self.fc_proj = None

    def forward(
        self,
        x: torch.Tensor,              # (B, L, embed_dim) token embeddings
        padding_mask: torch.Tensor,    # (B, L) True where padded
    ) -> torch.Tensor:
        """Encode route → z ∈ R^latent_dim.

        Returns: z (B, latent_dim)
        """
        x = self.pos_enc(x)
        src_key_padding_mask = padding_mask  # (B, L), True = ignore

        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # h: (B, L, embed_dim)

        # Mean pool over non-padded positions
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (B, L, 1)
        h_sum = (h * mask_expanded).sum(dim=1)  # (B, embed_dim)
        h_count = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        z = h_sum / h_count  # (B, embed_dim)

        if self.fc_proj is not None:
            z = self.fc_proj(z)

        return z

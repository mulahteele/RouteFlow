"""MLP velocity field v_θ(z_t, t) for improvement flow matching."""

from __future__ import annotations

import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    """MLP velocity network for flow matching.

    Architecture:
        input = concat(z_t, time_embed(t))  → (latent_dim + time_embed_dim)
        → Linear → SELU → Linear → SELU → ... → Linear → latent_dim

    Predicts velocity v(z_t, t) ∈ R^latent_dim.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding: scalar t → R^time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # MLP layers
        input_dim = latent_dim + time_embed_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity at (z_t, t).

        Args:
            z_t: (B, latent_dim) current position in latent space
            t: (B, 1) or (B,) time in [0, 1]

        Returns: (B, latent_dim) velocity vector
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        t_emb = self.time_embed(t)       # (B, time_embed_dim)
        x = torch.cat([z_t, t_emb], dim=-1)  # (B, latent_dim + time_embed_dim)
        return self.mlp(x)               # (B, latent_dim)

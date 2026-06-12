"""Online Reward-Weighted CFM with W2 regularization (ORW-CFM-W2).

Stage B: Fine-tune flow model with reward weighting and W2 divergence control.
Adapted from orw-cfm-main/finetune_fm.py for RouteFlow latent space.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from routeflow.models.velocity_net import VelocityNet
from routeflow.inference.optimize import ode_integrate


class ORWCFMTrainer:
    """Online Reward-Weighted CFM-W2 trainer for latent space flow matching.

    Each iteration:
      1. Sample z₁ from current model (z₀ ~ N(0,I) → ODE → z₁)
      2. Decode z₁ → molecules, evaluate via oracle → rewards
      3. Compute reward weights w = exp(τ * r)
      4. Update model with ORW-CFM-W2 loss:
         L = E[w * ||v_θ(z_t, t) - u_t||² + α * ||v_θ(z_t, t) - v_ref(z_t, t)||²]
    """

    def __init__(
        self,
        model: VelocityNet,
        lr: float = 2e-4,
        warmup_steps: int = 500,
        w2_coefficient: float = 1.0,
        temperature: float = 1.0,
        grad_clip: float = 1.0,
        device: str = "cuda",
        beta: float = 1.0,
    ):
        self.device = device
        self.alpha = w2_coefficient
        self.tau = temperature
        self.beta = beta
        self.grad_clip = grad_clip

        # Three copies of the model
        self.net_model = model.to(device)
        self.ref_model = copy.deepcopy(model).to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Optimizer and warmup scheduler
        self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=lr)
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step + 1, warmup_steps) / warmup_steps
        )

    @torch.no_grad()
    def sample(self, num_samples: int, ode_steps: int = 100) -> torch.Tensor:
        """Sample z₁ from current model via ODE integration.

        z₀ ~ N(0, I) → Euler ODE with v_θ → z₁
        Returns: (num_samples, latent_dim)
        """
        self.net_model.eval()
        latent_dim = self.net_model.latent_dim
        z0 = torch.randn(num_samples, latent_dim, device=self.device)

        trajectory = ode_integrate(
            self.net_model, z0,
            num_steps=ode_steps, endpoint=1.0, device=self.device,
        )
        z1 = trajectory[-1].to(self.device)
        return z1

    def compute_loss(
        self,
        z1: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute ORW-CFM-W2 loss.

        L = E[ w(z₁) * ||v_θ(z_t, t) - (z₁ - z₀)||²
              + α * ||v_θ(z_t, t) - v_ref(z_t, t)||² ]
        """
        B, D = z1.shape

        # Sample noise and time
        z0 = torch.randn_like(z1)
        t = torch.rand(B, 1, device=z1.device)

        # OT-path interpolation
        z_t = (1 - t) * z0 + t * z1
        target_v = z1 - z0

        # Predict velocity from current and reference models
        self.net_model.train()
        pred_v = self.net_model(z_t, t)

        with torch.no_grad():
            ref_v = self.ref_model(z_t, t)

        # FM loss per sample
        fm_loss = ((pred_v - target_v) ** 2).sum(dim=-1)    # (B,)

        # W2 regularization loss per sample
        w2_loss = ((pred_v - ref_v) ** 2).sum(dim=-1)       # (B,)

        # Combined loss: reward-weighted FM + W2
        total_loss = torch.mean(weights * fm_loss + self.alpha * w2_loss)

        metrics = {
            "fm_loss": fm_loss.mean().item(),
            "w2_loss": w2_loss.mean().item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def train_on_batch(
        self,
        z1: torch.Tensor,
        rewards: torch.Tensor,
        cycle_errors: torch.Tensor = None,
        train_steps: int = 10,
    ) -> dict:
        """Run multiple gradient steps on a single batch of (z₁, reward) pairs.

        Args:
            z1: (B, D) latent samples from the current model
            rewards: (B,) oracle rewards for each sample
            cycle_errors: (B,) optional ||z - E(D(z))||₂ per sample. If provided,
                weights become exp(τ * r_norm − β * e_norm) with both terms z-scored
                within batch so τ and β are dimensionless.
            train_steps: number of gradient steps on this batch

        Returns: averaged metrics dict
        """
        rewards = rewards.to(self.device)
        if cycle_errors is None or self.beta == 0.0:
            # Pure reward weighting (original behavior)
            weights = torch.exp(self.tau * rewards)
            cyc_mean = float("nan")
        else:
            cycle_errors = cycle_errors.to(self.device)
            # Z-score normalize both signals so τ and β share the same scale
            r_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            e_norm = (cycle_errors - cycle_errors.mean()) / (cycle_errors.std() + 1e-6)
            weights = torch.exp(self.tau * r_norm - self.beta * e_norm)
            cyc_mean = cycle_errors.mean().item()

        all_metrics = []
        for _ in range(train_steps):
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(z1, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net_model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0]
        }
        avg_metrics["mean_reward"] = rewards.mean().item()
        avg_metrics["max_reward"] = rewards.max().item()
        avg_metrics["mean_weight"] = weights.mean().item()
        avg_metrics["mean_cyc_err"] = cyc_mean
        return avg_metrics

    def save_checkpoint(self, path: str, iteration: int, best_reward: float):
        """Save training checkpoint."""
        torch.save({
            "net_model": self.net_model.state_dict(),
            "ref_model": self.ref_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "iteration": iteration,
            "best_reward": best_reward,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net_model.load_state_dict(ckpt["net_model"])
        self.ref_model.load_state_dict(ckpt["ref_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        return ckpt.get("iteration", 0), ckpt.get("best_reward", 0.0)

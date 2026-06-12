"""Conditional Flow Matching (CFM) pre-training on latent embeddings.

Stage A: Train v_θ_ref to map N(0,I) → data distribution q(z₁).
Uses linear OT-path interpolation: z_t = (1-t)*z₀ + t*z₁, target = z₁ - z₀.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from routeflow.models.velocity_net import VelocityNet


class LatentDataset(Dataset):
    """Dataset of latent embeddings z₁ for CFM training."""

    def __init__(self, latent_codes: np.ndarray):
        self.z = torch.from_numpy(latent_codes).float()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx]


def cfm_loss(model: VelocityNet, z1: torch.Tensor) -> torch.Tensor:
    """Compute CFM loss for a batch of target embeddings z₁.

    L = E_t,z₁ [ ||v_θ(z_t, t) - (z₁ - z₀)||² ]
    where z₀ ~ N(0,I), t ~ U(0,1), z_t = (1-t)*z₀ + t*z₁.
    """
    B, D = z1.shape
    device = z1.device

    # Sample noise and time
    z0 = torch.randn_like(z1)                       # (B, D)
    t = torch.rand(B, 1, device=device)              # (B, 1)

    # OT-path interpolation
    z_t = (1 - t) * z0 + t * z1                      # (B, D)

    # Target velocity (constant for linear path)
    target_v = z1 - z0                                # (B, D)

    # Predict velocity
    pred_v = model(z_t, t)                            # (B, D)

    # MSE loss
    loss = ((pred_v - target_v) ** 2).sum(dim=-1).mean()
    return loss


def train_cfm(
    latent_codes: np.ndarray,
    val_latent_codes: np.ndarray = None,
    latent_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    time_embed_dim: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 500,
    batch_size: int = 256,
    patience: int = 30,
    grad_clip: float = 1.0,
    device: str = "cuda",
    save_path: str = "checkpoints/flow_pretrain_best.pt",
) -> VelocityNet:
    """Train CFM model: N(0,I) → q(z₁).

    Args:
        latent_codes: training z embeddings
        val_latent_codes: validation z embeddings (if None, uses 90/10 split from train)
    """
    if val_latent_codes is not None:
        train_dataset = LatentDataset(latent_codes)
        val_dataset = LatentDataset(val_latent_codes)
    else:
        # Fallback: 90/10 split
        N = len(latent_codes)
        rng = np.random.RandomState(42)
        perm = rng.permutation(N)
        val_size = max(1, int(0.1 * N))
        train_dataset = LatentDataset(latent_codes[perm[val_size:]])
        val_dataset = LatentDataset(latent_codes[perm[:val_size]])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = VelocityNet(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for z1_batch in pbar:
            z1_batch = z1_batch.to(device)
            B = z1_batch.shape[0]

            loss = cfm_loss(model, z1_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss_sum += loss.item() * B
            train_count += B
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / max(train_count, 1)

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for z1_batch in val_loader:
                z1_batch = z1_batch.to(device)
                B = z1_batch.shape[0]
                loss = cfm_loss(model, z1_batch)
                val_loss_sum += loss.item() * B
                val_count += B

        val_loss = val_loss_sum / max(val_count, 1)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, save_path)
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Load best
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nCFM training complete. Best val_loss={best_val_loss:.6f}")
    return model

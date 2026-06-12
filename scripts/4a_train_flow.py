"""Phase 3A: Pre-train flow matching model (CFM) on latent embeddings.

Encode all 100K training routes with frozen autoencoder → z₁ embeddings.
Train v_θ_ref: N(0,I) → q(z₁) via Conditional Flow Matching.
Zero oracle calls.
"""

import os
import pickle
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from routeflow.data.route_dataset import RouteDataset
from routeflow.data.collate import collate_routes
from routeflow.models.autoencoder import RouteAutoencoder
from routeflow.flow.cfm import train_cfm


def main():
    parser = argparse.ArgumentParser(description="Phase 3A: Pre-train flow (CFM)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    ae_cfg = cfg["autoencoder"]
    flow_cfg = cfg["flow_pretrain"]
    precomp = cfg["precompute"]
    out_dir = paths["processed_dir"]
    ckpt_dir = paths["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ae_tag = cfg.get("ae_tag", "")
    tag_suffix = f"_{ae_tag}" if ae_tag else ""
    print(f"AE tag: {ae_tag or '(none)'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load data ----
    print("Loading data...")
    bb_fps = np.load(os.path.join(out_dir, "bb_fingerprints.npy"))

    with open(os.path.join(out_dir, "routes_train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    train_routes = train_data["routes"]

    with open(os.path.join(out_dir, "routes_val.pkl"), "rb") as f:
        val_data = pickle.load(f)
    val_routes = val_data["routes"]

    with open(os.path.join(out_dir, "compatibility_matrix.pkl"), "rb") as f:
        compat_data = pickle.load(f)
    num_rxn = len(compat_data["reactions"])

    print(f"  {len(train_routes)} training routes, {len(val_routes)} val routes, {num_rxn} reactions")

    # ---- Load frozen autoencoder ----
    print("Loading autoencoder...")
    ae_model = RouteAutoencoder(
        num_rxn=num_rxn,
        bb_fingerprints=bb_fps,
        embed_dim=ae_cfg["embed_dim"],
        latent_dim=ae_cfg["latent_dim"],
        encoder_num_layers=ae_cfg["encoder_num_layers"],
        encoder_nhead=ae_cfg["encoder_nhead"],
        encoder_ff_dim=ae_cfg["encoder_ff_dim"],
        encoder_dropout=ae_cfg["encoder_dropout"],
        decoder_num_layers=ae_cfg["decoder_num_layers"],
        decoder_nhead=ae_cfg["decoder_nhead"],
        decoder_ff_dim=ae_cfg["decoder_ff_dim"],
        decoder_dropout=ae_cfg["decoder_dropout"],
        max_len=ae_cfg["decoder_max_len"],
        fp_nbits=precomp["fp_nbits"],
    ).to(device)

    ckpt = torch.load(
        os.path.join(ckpt_dir, f"autoencoder{tag_suffix}_best.pt"),
        map_location=device, weights_only=False,
    )
    ae_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ae_model.eval()
    print(f"  Loaded autoencoder from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # ---- Encode all training routes ----
    print("Encoding all training routes...")
    dataset = RouteDataset(train_routes, bb_fps)
    loader = DataLoader(
        dataset, batch_size=ae_cfg["batch_size"], shuffle=False,
        collate_fn=collate_routes, num_workers=4,
    )

    all_z = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            z = ae_model.encode(
                batch_dev["token_types"],
                batch_dev["token_values"],
                batch_dev["padding_mask"],
            )
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy()  # (N, latent_dim)
    print(f"  Encoded {all_z.shape[0]} routes → z ∈ R^{all_z.shape[1]}")

    # Save embeddings for later use
    z_path = os.path.join(out_dir, f"train_embeddings{tag_suffix}.npy")
    np.save(z_path, all_z)
    print(f"  Saved embeddings to {z_path}")

    # Print distribution stats
    print(f"  z stats: mean={all_z.mean():.4f}, std={all_z.std():.4f}, "
          f"min={all_z.min():.4f}, max={all_z.max():.4f}")
    per_dim_std = all_z.std(axis=0)
    print(f"  Per-dim std: mean={per_dim_std.mean():.4f}, "
          f"min={per_dim_std.min():.4f}, max={per_dim_std.max():.4f}")

    # ---- Encode val routes ----
    print("Encoding val routes...")
    val_dataset = RouteDataset(val_routes, bb_fps)
    val_loader = DataLoader(
        val_dataset, batch_size=ae_cfg["batch_size"], shuffle=False,
        collate_fn=collate_routes, num_workers=4,
    )
    val_z = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Encoding val"):
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            z = ae_model.encode(
                batch_dev["token_types"], batch_dev["token_values"], batch_dev["padding_mask"],
            )
            val_z.append(z.cpu())
    val_z = torch.cat(val_z, dim=0).numpy()
    print(f"  Encoded {val_z.shape[0]} val routes")

    # ---- Train CFM (no normalization — use raw z) ----
    print("\nTraining CFM (Stage A)...")
    save_path = os.path.join(ckpt_dir, f"flow_pretrain{tag_suffix}_best.pt")

    model = train_cfm(
        latent_codes=all_z,
        val_latent_codes=val_z,
        latent_dim=ae_cfg["latent_dim"],
        hidden_dim=flow_cfg["hidden_dim"],
        num_layers=flow_cfg["num_layers"],
        time_embed_dim=flow_cfg["time_embed_dim"],
        lr=flow_cfg["lr"],
        weight_decay=flow_cfg["weight_decay"],
        epochs=flow_cfg["epochs"],
        batch_size=flow_cfg["batch_size"],
        patience=flow_cfg["patience"],
        grad_clip=flow_cfg["grad_clip"],
        device=device,
        save_path=save_path,
    )

    print(f"\nPhase 3A complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()

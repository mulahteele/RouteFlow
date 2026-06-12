"""Phase 2: Train route autoencoder (supports single-GPU and multi-GPU DDP)."""

import os
import sys
import pickle
import argparse

import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from routeflow.data.route_dataset import RouteDataset
from routeflow.data.collate import collate_routes
from routeflow.models.autoencoder import RouteAutoencoder


def is_distributed():
    return dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train autoencoder")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # --- DDP init ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    ae_cfg = cfg["autoencoder"]
    precomp = cfg["precompute"]
    out_dir = paths["processed_dir"]
    ckpt_dir = paths["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    if is_main_process():
        print(f"Using device: {device}  (world_size={world_size})")

    # Load data
    if is_main_process():
        print("Loading data...")
    bb_fps = np.load(os.path.join(out_dir, "bb_fingerprints.npy"))
    if is_main_process():
        print(f"  BB fingerprints: {bb_fps.shape}")

    with open(os.path.join(out_dir, "routes_train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(out_dir, "routes_val.pkl"), "rb") as f:
        val_data = pickle.load(f)

    train_routes = train_data["routes"]
    val_routes = val_data["routes"]

    if is_main_process():
        print(f"  Train: {len(train_routes)} routes")
        print(f"  Val: {len(val_routes)} routes")

    # Load compatibility matrix
    with open(os.path.join(out_dir, "compatibility_matrix.pkl"), "rb") as f:
        compat_data = pickle.load(f)
    num_rxn = len(compat_data["reactions"])
    if is_main_process():
        print(f"  Num reactions: {num_rxn}")

    from routeflow.chem.compatibility import CompatibilityMatrix
    from rdkit import Chem
    bb_smiles = compat_data["bb_smiles"]
    bb_mols = [Chem.MolFromSmiles(s) for s in bb_smiles]
    compat = CompatibilityMatrix(
        bb_smiles, bb_mols, compat_data["reactions"], matrix=compat_data["matrix"]
    )

    # Create datasets
    train_dataset = RouteDataset(train_routes, bb_fps, compat_matrix=compat)
    val_dataset = RouteDataset(val_routes, bb_fps, compat_matrix=compat)

    # Samplers for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=ae_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_routes,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ae_cfg["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_routes,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Create model
    model = RouteAutoencoder(
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

    if is_distributed():
        # find_unused_parameters=True: embedding.pop_emb is only used when a
        # batch contains POP tokens. Some batches under the new (filtered) data
        # distribution have no POP, leaving pop_emb without a gradient and
        # tripping DDP's all-reduce sync check otherwise.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # For accessing underlying model (DDP wraps it)
    raw_model = model.module if is_distributed() else model

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ae_cfg["lr"],
        weight_decay=ae_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    ae_tag = cfg.get("ae_tag", "")
    tag_suffix = f"_{ae_tag}" if ae_tag else ""
    save_path = os.path.join(ckpt_dir, f"autoencoder{tag_suffix}_best.pt")
    if is_main_process():
        print(f"AE tag: {ae_tag or '(none)'}  → checkpoint: {save_path}")

    # Noise injection schedule
    noise_sigma_target = ae_cfg.get("noise_sigma_target", 0.0)
    noise_warmup_epochs = ae_cfg.get("noise_warmup_epochs", 50)

    # Scheduled sampling schedule
    ss_max = ae_cfg.get("scheduled_sampling_max", 0.0)
    ss_warmup = ae_cfg.get("scheduled_sampling_warmup", 50)

    # Resume from checkpoint if specified
    resume_path = ae_cfg.get("resume_from", None)
    start_epoch = 0
    if resume_path and os.path.exists(os.path.join(ckpt_dir, resume_path)):
        ckpt_resume = torch.load(
            os.path.join(ckpt_dir, resume_path),
            map_location=device, weights_only=False,
        )
        raw_model.load_state_dict(ckpt_resume["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt_resume:
            optimizer.load_state_dict(ckpt_resume["optimizer_state_dict"])
        start_epoch = ckpt_resume.get("epoch", 0) + 1
        best_val_loss = ckpt_resume.get("val_loss", float("inf"))
        if is_main_process():
            print(f"Resumed from {resume_path} (epoch {start_epoch}, val_loss={best_val_loss:.4f})")

    for epoch in range(start_epoch, ae_cfg["epochs"]):
        # Set epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Compute current noise sigma (linear anneal from 0 to target)
        if noise_sigma_target > 0 and noise_warmup_epochs > 0:
            noise_sigma = noise_sigma_target * min(1.0, epoch / noise_warmup_epochs)
        else:
            noise_sigma = 0.0

        # Compute scheduled sampling probability (linear anneal from 0 to max)
        if ss_max > 0 and ss_warmup > 0:
            ss_p = ss_max * min(1.0, epoch / ss_warmup)
        else:
            ss_p = 0.0

        # ---- Train ----
        model.train()
        train_losses = {"loss": 0, "loss_action": 0, "loss_rxn": 0, "loss_source": 0}
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{ae_cfg['epochs']} [train]",
                     disable=not is_main_process() or not sys.stdout.isatty())
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            losses = model(batch, noise_sigma=noise_sigma, scheduled_sampling_p=ss_p)
            loss = losses["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ae_cfg["grad_clip"])
            optimizer.step()

            B = batch["seq_lens"].shape[0]
            for k in train_losses:
                train_losses[k] += losses[k].item() * B
            train_count += B

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                act=f"{losses['loss_action'].item():.4f}",
                rxn=f"{losses['loss_rxn'].item():.4f}",
                src=f"{losses['loss_source'].item():.4f}",
            )

        for k in train_losses:
            train_losses[k] /= max(train_count, 1)

        # ---- Validate ----
        model.eval()
        val_losses = {"loss": 0, "loss_action": 0, "loss_rxn": 0, "loss_source": 0}
        val_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", leave=False,
                              disable=not is_main_process() or not sys.stdout.isatty()):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                losses = model(batch)
                B = batch["seq_lens"].shape[0]
                for k in val_losses:
                    val_losses[k] += losses[k].item() * B
                val_count += B

        # Aggregate val losses across all ranks
        if is_distributed():
            for k in val_losses:
                t = torch.tensor([val_losses[k], val_count], device=device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                val_losses[k] = t[0].item()
            val_count = int(t[1].item())

        for k in val_losses:
            val_losses[k] /= max(val_count, 1)

        scheduler.step(val_losses["loss"])

        if is_main_process():
            extra = ""
            if noise_sigma_target > 0:
                extra += f"  σ={noise_sigma:.3f}"
            if ss_max > 0:
                extra += f"  ss={ss_p:.3f}"
            print(f"Epoch {epoch+1}/{ae_cfg['epochs']}  "
                  f"train_loss={train_losses['loss']:.4f}  "
                  f"val_loss={val_losses['loss']:.4f}  "
                  f"[act={val_losses['loss_action']:.4f} "
                  f"rxn={val_losses['loss_rxn']:.4f} "
                  f"src={val_losses['loss_source']:.4f}]  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{extra}")

            # Early stopping & checkpoint (only rank 0 saves)
            if val_losses["loss"] < best_val_loss:
                best_val_loss = val_losses["loss"]
                epochs_no_improve = 0
                torch.save({
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "config": ae_cfg,
                }, save_path)
                print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

        # Broadcast early stopping decision from rank 0
        if is_distributed():
            stop_tensor = torch.tensor([epochs_no_improve], device=device)
            dist.broadcast(stop_tensor, src=0)
            epochs_no_improve = stop_tensor.item()

        if epochs_no_improve >= ae_cfg["patience"]:
            if is_main_process():
                print(f"Early stopping after {epoch+1} epochs")
            break

    if is_main_process():
        print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
        print(f"Best model saved to {save_path}")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

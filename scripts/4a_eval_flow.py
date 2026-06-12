"""Phase 3A eval: Evaluate pre-trained flow model (CFM).

Sample z₁ from v_θ_ref, decode via frozen autoencoder, measure:
- Decode validity, uniqueness, distribution match.
"""

import os
import pickle
import argparse

import yaml
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from routeflow.models.autoencoder import RouteAutoencoder
from routeflow.models.velocity_net import VelocityNet
from routeflow.chem.compatibility import CompatibilityMatrix
from routeflow.inference.optimize import ode_integrate


def main():
    parser = argparse.ArgumentParser(description="Phase 3A eval: Evaluate pre-trained flow")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    ae_cfg = cfg["autoencoder"]
    flow_cfg = cfg["flow_pretrain"]
    precomp = cfg["precompute"]
    out_dir = paths["processed_dir"]
    ckpt_dir = paths["checkpoint_dir"]
    ae_tag = cfg.get("ae_tag", "")
    tag_suffix = f"_{ae_tag}" if ae_tag else ""
    print(f"AE tag: {ae_tag or '(none)'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load compatibility matrix + BB data ----
    print("Loading compatibility matrix...")
    with open(os.path.join(out_dir, "compatibility_matrix.pkl"), "rb") as f:
        compat_data = pickle.load(f)
    bb_smiles = compat_data["bb_smiles"]
    bb_mols = [Chem.MolFromSmiles(s) for s in bb_smiles]
    compat = CompatibilityMatrix(
        bb_smiles, bb_mols, compat_data["reactions"], matrix=compat_data["matrix"]
    )
    num_rxn = len(compat_data["reactions"])
    bb_fps = np.load(os.path.join(out_dir, "bb_fingerprints.npy"))

    # ---- Load autoencoder ----
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

    # ---- Load flow model ----
    print("Loading pre-trained flow model...")
    flow_model = VelocityNet(
        latent_dim=ae_cfg["latent_dim"],
        hidden_dim=flow_cfg["hidden_dim"],
        num_layers=flow_cfg["num_layers"],
        time_embed_dim=flow_cfg["time_embed_dim"],
    ).to(device)

    flow_ckpt = torch.load(
        os.path.join(ckpt_dir, f"flow_pretrain{tag_suffix}_best.pt"),
        map_location=device, weights_only=False,
    )
    flow_model.load_state_dict(flow_ckpt["model_state_dict"])
    flow_model.eval()

    # ---- Sample z₁ from flow model ----
    print(f"Sampling {args.num_samples} embeddings from flow model...")
    z0 = torch.randn(args.num_samples, ae_cfg["latent_dim"], device=device)
    trajectory = ode_integrate(flow_model, z0, num_steps=30, endpoint=1.0, device=device)
    z1 = trajectory[-1].to(device)

    # ---- Distribution match ----
    print("\nDistribution statistics (generated z₁):")
    z1_np = z1.cpu().numpy()
    print(f"  mean={z1_np.mean():.4f}, std={z1_np.std():.4f}")

    # Compare with training embeddings
    train_z_path = os.path.join(out_dir, f"train_embeddings{tag_suffix}.npy")
    if os.path.exists(train_z_path):
        train_z = np.load(train_z_path)
        print(f"Training z₁ stats: mean={train_z.mean():.4f}, std={train_z.std():.4f}")
        # Per-dimension comparison
        gen_mean = z1_np.mean(axis=0)
        train_mean = train_z.mean(axis=0)
        gen_std = z1_np.std(axis=0)
        train_std = train_z.std(axis=0)
        mean_mae = np.abs(gen_mean - train_mean).mean()
        std_mae = np.abs(gen_std - train_std).mean()
        print(f"  Per-dim mean MAE: {mean_mae:.4f}")
        print(f"  Per-dim std MAE:  {std_mae:.4f}")

    # ---- Decode and evaluate ----
    print(f"\nDecoding {args.num_samples} samples...")
    smiles_list = ae_model.decode_z_to_smiles(
        z1, compat, bb_mols, beam_width=5,
    )

    # Validity
    n_valid = sum(1 for s in smiles_list if s is not None)
    validity = n_valid / args.num_samples

    # Uniqueness
    valid_smiles = [s for s in smiles_list if s is not None]
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / max(len(valid_smiles), 1)

    print(f"\n{'='*50}")
    print(f"  Flow Pre-training Evaluation")
    print(f"{'='*50}")
    print(f"  {'Samples':<25s} {args.num_samples}")
    print(f"  {'Decode validity':<25s} {validity*100:6.2f}%")
    print(f"  {'Uniqueness':<25s} {uniqueness*100:6.2f}%")
    print(f"  {'Valid molecules':<25s} {n_valid}")
    print(f"  {'Unique molecules':<25s} {len(unique_smiles)}")
    print(f"{'='*50}\n")

    # Save results
    save_path = os.path.join(out_dir, "flow_pretrain_eval.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "validity": validity,
            "uniqueness": uniqueness,
            "n_valid": n_valid,
            "n_unique": len(unique_smiles),
            "smiles": smiles_list,
            "z1": z1_np,
        }, f)
    print(f"Saved eval results to {save_path}")


if __name__ == "__main__":
    main()

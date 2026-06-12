"""Phase 2 eval: Evaluate autoencoder reconstruction via closed-loop decoding."""

import os
import pickle
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from tqdm import tqdm

from routeflow.data.route_dataset import RouteDataset
from routeflow.data.collate import collate_routes
from routeflow.models.autoencoder import RouteAutoencoder
from routeflow.chem.compatibility import CompatibilityMatrix
from routeflow.utils.metrics import compute_ae_metrics, print_ae_metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Evaluate autoencoder")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="val",
                        help="Which split to evaluate: val / test / test_hard "
                             "(or any other suffix matching routes_<split>.pkl).")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of routes to evaluate (default: all)")
    parser.add_argument("--noise_robustness", action="store_true",
                        help="Run noise robustness curve experiment")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    ae_cfg = cfg["autoencoder"]
    precomp = cfg["precompute"]
    out_dir = paths["processed_dir"]
    ckpt_dir = paths["checkpoint_dir"]
    ae_tag = cfg.get("ae_tag", "")
    tag_suffix = f"_{ae_tag}" if ae_tag else ""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("Loading data...")
    bb_fps = np.load(os.path.join(out_dir, "bb_fingerprints.npy"))
    with open(os.path.join(out_dir, f"routes_{args.split}.pkl"), "rb") as f:
        data = pickle.load(f)
    routes = data["routes"]

    # Load compatibility matrix
    with open(os.path.join(out_dir, "compatibility_matrix.pkl"), "rb") as f:
        compat_data = pickle.load(f)
    num_rxn = len(compat_data["reactions"])
    bb_smiles = compat_data["bb_smiles"]
    bb_mols = [Chem.MolFromSmiles(s) for s in bb_smiles]
    compat = CompatibilityMatrix(
        bb_smiles, bb_mols, compat_data["reactions"], matrix=compat_data["matrix"]
    )

    # Load model
    print("Loading model...")
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

    ckpt_path = os.path.join(ckpt_dir, f"autoencoder{tag_suffix}_best.pt")
    print(f"AE tag: {ae_tag or '(none)'}  → loading {ckpt_path}")
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # ---- Step 1: Compute val/test loss ----
    print("\nComputing loss...")
    dataset = RouteDataset(routes, bb_fps, compat_matrix=compat)
    loader = DataLoader(
        dataset,
        batch_size=ae_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_routes,
        num_workers=4,
    )

    total_losses = {"loss": 0, "loss_action": 0, "loss_rxn": 0, "loss_source": 0}
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing loss"):
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            losses = model(batch_dev)
            B = batch["seq_lens"].shape[0]
            for k in total_losses:
                total_losses[k] += losses[k].item() * B
            total_count += B

    for k in total_losses:
        total_losses[k] /= max(total_count, 1)

    print(f"\n{args.split} loss: {total_losses['loss']:.4f}")
    print(f"  action: {total_losses['loss_action']:.4f}")
    print(f"  rxn:    {total_losses['loss_rxn']:.4f}")
    print(f"  source: {total_losses['loss_source']:.4f}")

    # ---- Step 2: Closed-loop reconstruction ----
    print(f"\nRunning closed-loop reconstruction on {args.split} set...")
    eval_routes = routes
    if args.num_samples is not None:
        eval_routes = routes[:args.num_samples]

    # Encode all routes to z
    eval_dataset = RouteDataset(eval_routes, bb_fps)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=ae_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_routes,
        num_workers=4,
    )

    all_z = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Encoding"):
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            z = model.encode(
                batch_dev["token_types"],
                batch_dev["token_values"],
                batch_dev["padding_mask"],
            )
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)  # (N, latent_dim)

    # Decode each z back to route via closed-loop (greedy)
    reconstructed = []
    for i in tqdm(range(len(eval_routes)), desc="Closed-loop decoding (greedy)"):
        z_i = all_z[i:i+1].to(device)
        recon = model.decode_autoregressive(
            z_i, compat, bb_mols, greedy=True,
        )
        reconstructed.append(recon)

    # Compute metrics
    metrics = compute_ae_metrics(eval_routes, reconstructed)
    print_ae_metrics(metrics, split=args.split)

    # Save results
    save_path = os.path.join(out_dir, f"ae_eval{tag_suffix}_{args.split}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "metrics": metrics,
            "losses": total_losses,
            "num_samples": len(eval_routes),
        }, f)
    print(f"Saved eval results to {save_path}")

    # Save per-sample (original, reconstructed) SMILES pairs to autoencoder_result/.
    # Mapping: test_easy → routeflow_easy.txt, test_hard → routeflow_hard.txt;
    # any other split keeps its name (routeflow_<split>.txt).
    split_short = args.split.replace("test_", "") if args.split.startswith("test_") else args.split
    pair_dir = "autoencoder_result"
    os.makedirs(pair_dir, exist_ok=True)
    pair_path = os.path.join(pair_dir, f"routeflow_{split_short}.txt")
    with open(pair_path, "w") as f:
        f.write("idx\toriginal_smiles\treconstructed_smiles\n")
        for i, (orig, recon) in enumerate(zip(eval_routes, reconstructed)):
            orig_smi = getattr(orig, "final_product", "") or ""
            if orig_smi:
                mol = Chem.MolFromSmiles(orig_smi)
                if mol is not None:
                    orig_smi = Chem.MolToSmiles(mol)
            if recon is None:
                recon_smi = ""
            else:
                recon_smi = getattr(recon, "final_product", "") or ""
                if recon_smi:
                    mol = Chem.MolFromSmiles(recon_smi)
                    if mol is not None:
                        recon_smi = Chem.MolToSmiles(mol)
            f.write(f"{i}\t{orig_smi}\t{recon_smi}\n")
    print(f"Saved SMILES pairs to {pair_path}")

    # ---- Step 3 (optional): Noise robustness curve ----
    if args.noise_robustness:
        sigma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        num_eval = min(len(eval_routes), 500)  # cap at 500 for speed

        # Use pre-computed z from step 2
        eval_z = all_z[:num_eval]

        print(f"\n{'='*60}")
        print(f"  Noise Robustness Curve ({args.split}, {num_eval} routes)")
        print(f"{'='*60}")
        print(f"  {'σ':>6s}  {'Validity':>10s}  {'Product Match':>14s}  {'Tanimoto':>10s}")
        print(f"  {'-'*44}")

        robustness_results = []
        for sigma in sigma_values:
            # Add noise to z
            if sigma > 0:
                noise = torch.randn_like(eval_z) * sigma
                z_noisy = eval_z + noise
            else:
                z_noisy = eval_z

            # Decode each noisy z
            recon_noisy = []
            for i in range(num_eval):
                z_i = z_noisy[i:i+1].to(device)
                recon = model.decode_autoregressive(
                    z_i, compat, bb_mols, greedy=True,
                )
                recon_noisy.append(recon)

            # Compute metrics
            m = compute_ae_metrics(eval_routes[:num_eval], recon_noisy)
            robustness_results.append({
                "sigma": sigma,
                "validity": m["validity"],
                "product_match": m["product_match"],
                "tanimoto_sim_all": m["tanimoto_sim_all"],
            })

            print(f"  {sigma:6.1f}  {m['validity']*100:9.1f}%  "
                  f"{m['product_match']*100:13.1f}%  "
                  f"{m['tanimoto_sim_all']:10.4f}")

        print(f"{'='*60}\n")

        # Save robustness results
        rob_path = os.path.join(out_dir, f"ae_noise_robustness{tag_suffix}_{args.split}.pkl")
        with open(rob_path, "wb") as f:
            pickle.dump(robustness_results, f)
        print(f"Saved noise robustness results to {rob_path}")


if __name__ == "__main__":
    main()

"""Phase 3B: Online Reward-Weighted Flow Fine-tuning (ORW-CFM-W2).

100 iterations × 100 samples/iter = 10K oracle calls.
Each iteration: sample → decode → oracle → reward-weighted update.
"""

import os
import pickle
import argparse
import json

import yaml
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from routeflow.models.autoencoder import RouteAutoencoder
from routeflow.models.velocity_net import VelocityNet
from routeflow.chem.compatibility import CompatibilityMatrix
from routeflow.flow.orw_cfm import ORWCFMTrainer


@torch.no_grad()
def encode_routes(ae_model, routes, device):
    """Re-encode a list of decoded Route objects to latent vectors.

    Returns (B, latent_dim) tensor of E(D(z)) embeddings. Manually pads to
    the longest sequence in the batch; does not need source_candidates because
    encode() only uses token_types/values/padding_mask.
    """
    if not routes:
        return torch.empty(0, ae_model.latent_dim, device=device)

    seq_lens = [len(r.token_types()) for r in routes]
    max_len = max(seq_lens)
    B = len(routes)
    token_types = torch.full((B, max_len), -1, dtype=torch.long, device=device)
    token_values = torch.zeros((B, max_len), dtype=torch.long, device=device)
    padding_mask = torch.ones((B, max_len), dtype=torch.bool, device=device)

    for i, r in enumerate(routes):
        tt = r.token_types()
        tv = r.token_values()
        sl = len(tt)
        token_types[i, :sl] = torch.tensor(tt, dtype=torch.long, device=device)
        token_values[i, :sl] = torch.tensor(tv, dtype=torch.long, device=device)
        padding_mask[i, :sl] = False

    return ae_model.encode(token_types, token_values, padding_mask)


def internal_diversity(smiles_list):
    """Internal Diversity = 1 - mean pairwise Tanimoto similarity.

    Uses Morgan fingerprints (radius=2, 2048 bits). Duplicate SMILES yield
    Tanimoto=1.0 between themselves, lowering the diversity score. Range: [0, 1].
    """
    if not smiles_list or len(smiles_list) < 2:
        return 0.0
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    if not sims:
        return 0.0
    return 1.0 - (sum(sims) / len(sims))


def main():
    parser = argparse.ArgumentParser(description="Phase 3B: ORW-CFM-W2 fine-tuning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--property", type=str, default="GSK3B",
                        help="Oracle property name (e.g. GSK3B, amlodipine_mpo, DRD3)")
    parser.add_argument("--best_config", type=str, default=None,
                        help="YAML with per-property hyperparameters. Two supported "
                             "schemas: (1) per_property_best[<property>] = dict — "
                             "single config per property; (2) per_property_configs"
                             "[<property>] = list of dicts — pick variant via --variant.")
    parser.add_argument("--variant", type=int, default=None,
                        help="1-indexed variant to pick when --best_config uses "
                             "the per_property_configs (list) schema.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for torch / numpy / random "
                             "(also overrides ft_cfg.seed).")
    # Optional hyperparameter overrides for ablation
    parser.add_argument("--tau_min", type=float, default=None)
    parser.add_argument("--tau_max", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--replay_size", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--samples_per_iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ode_steps", type=int, default=None)
    parser.add_argument("--replay_sample_frac", type=float, default=None,
                        help="Fraction of replay buffer sampled per iteration "
                             "and mixed with the current batch (default 0.2).")
    parser.add_argument("--beta", type=float, default=None,
                        help="Cycle-consistency weight in reward shaping. "
                             "weights = exp(τ·r_norm − β·e_norm) with both "
                             "z-scored within batch. 0 disables (default).")
    parser.add_argument("--consistency_threshold", type=float, default=None,
                        help="Replay buffer admission threshold on e_cyc = "
                             "||z − E(D(z))||₂. Only samples with e_cyc < this "
                             "value enter the buffer. Default 1e-4 (≈exact match "
                             "modulo float32 noise).")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for ablation (used in output paths)")
    parser.add_argument("--no_save_checkpoint", action="store_true",
                        help="Skip saving _latest.pt and _best.pt (for ablation sweeps).")
    args = parser.parse_args()

    prop_name = args.property

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    ae_cfg = cfg["autoencoder"]
    flow_pre_cfg = cfg["flow_pretrain"]
    ft_cfg = cfg["flow_finetune"]

    # Apply per-property best YAML (lower priority than CLI flags below)
    if args.best_config is not None:
        with open(args.best_config) as f:
            best_yaml = yaml.safe_load(f)
        # Two supported schemas:
        #   per_property_best[<prop>]    -> dict (single config per property)
        #   per_property_configs[<prop>] -> list of dicts (use --variant to pick)
        bp = None
        src_label = ""
        if "per_property_best" in best_yaml and prop_name in best_yaml["per_property_best"]:
            bp = best_yaml["per_property_best"][prop_name]
            src_label = "per_property_best"
        elif "per_property_configs" in best_yaml and prop_name in best_yaml["per_property_configs"]:
            variants = best_yaml["per_property_configs"][prop_name]
            if args.variant is None:
                raise ValueError(
                    f"--variant is required when {args.best_config} uses "
                    f"per_property_configs (list schema)."
                )
            if args.variant < 1 or args.variant > len(variants):
                raise IndexError(
                    f"--variant {args.variant} out of range "
                    f"[1, {len(variants)}] for property {prop_name}."
                )
            bp = variants[args.variant - 1]
            src_label = f"per_property_configs[v{args.variant}/{len(variants)}]"
        else:
            raise KeyError(
                f"Property '{prop_name}' not found in {args.best_config} "
                f"under per_property_best or per_property_configs."
            )
        # Map YAML fields -> ft_cfg keys
        key_map = {
            "tau_min":          "temperature_min",
            "tau_max":          "temperature_max",
            "alpha":            "w2_coefficient",
            "beta":             "cycle_beta",
            "replay_size":      "replay_size",
            "num_iterations":   "num_iterations",
            "samples_per_iter": "samples_per_iter",
            "lr":               "lr",
            "ode_steps":        "ode_steps",
            "replay_sample_frac": "replay_sample_frac",
        }
        for src, dst in key_map.items():
            if src in bp:
                ft_cfg[dst] = bp[src]
        print(f"  [best_config] overrode ft_cfg from {args.best_config} "
              f"({src_label}) for {prop_name}")

    # Override config with CLI args if provided (highest priority)
    if args.tau_min is not None:
        ft_cfg["temperature_min"] = args.tau_min
    if args.tau_max is not None:
        ft_cfg["temperature_max"] = args.tau_max
    if args.alpha is not None:
        ft_cfg["w2_coefficient"] = args.alpha
    if args.replay_size is not None:
        ft_cfg["replay_size"] = args.replay_size
    if args.num_iterations is not None:
        ft_cfg["num_iterations"] = args.num_iterations
    if args.samples_per_iter is not None:
        ft_cfg["samples_per_iter"] = args.samples_per_iter
    if args.lr is not None:
        ft_cfg["lr"] = args.lr
    if args.ode_steps is not None:
        ft_cfg["ode_steps"] = args.ode_steps
    if args.replay_sample_frac is not None:
        ft_cfg["replay_sample_frac"] = args.replay_sample_frac
    if args.beta is not None:
        ft_cfg["cycle_beta"] = args.beta
    if args.consistency_threshold is not None:
        ft_cfg["consistency_threshold"] = args.consistency_threshold
    if args.seed is not None:
        ft_cfg["seed"] = args.seed

    # Set random seeds for reproducibility across torch / numpy / random
    seed = int(ft_cfg.get("seed", 42))
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  [seed] using seed={seed}")

    # Run name for output paths (default: property name)
    run_name = args.run_name if args.run_name else prop_name
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

    # ---- Load pre-trained flow model ----
    print("Loading pre-trained flow model (Stage A)...")
    flow_model = VelocityNet(
        latent_dim=ae_cfg["latent_dim"],
        hidden_dim=flow_pre_cfg["hidden_dim"],
        num_layers=flow_pre_cfg["num_layers"],
        time_embed_dim=flow_pre_cfg["time_embed_dim"],
    ).to(device)

    flow_ckpt = torch.load(
        os.path.join(ckpt_dir, f"flow_pretrain{tag_suffix}_best.pt"),
        map_location=device, weights_only=False,
    )
    flow_model.load_state_dict(flow_ckpt["model_state_dict"])

    # ---- Setup ORW-CFM-W2 trainer ----
    tau_min = ft_cfg["temperature_min"]
    tau_max = ft_cfg["temperature_max"]
    cycle_beta = float(ft_cfg.get("cycle_beta", 0.0))
    trainer = ORWCFMTrainer(
        model=flow_model,
        lr=ft_cfg["lr"],
        warmup_steps=ft_cfg["warmup_steps"],
        w2_coefficient=ft_cfg["w2_coefficient"],
        temperature=tau_min,  # will be annealed during training
        grad_clip=ft_cfg["grad_clip"],
        device=device,
        beta=cycle_beta,
    )

    # ---- Setup oracle ----
    from routeflow.oracle import get_oracle
    oracle = get_oracle(prop_name)
    print(f"  Oracle: {prop_name}")

    # ---- Online training loop ----
    num_iterations = ft_cfg["num_iterations"]
    samples_per_iter = ft_cfg["samples_per_iter"]
    train_steps = ft_cfg["train_steps_per_iter"]
    ode_steps = ft_cfg["ode_steps"]

    print(f"\nStarting ORW-CFM-W2 fine-tuning:")
    print(f"  {num_iterations} iterations × {samples_per_iter} samples")
    print(f"  α (W2 coeff) = {ft_cfg['w2_coefficient']}")
    print(f"  τ (temperature) = {tau_min} → {tau_max}")
    print(f"  β (cycle weight) = {cycle_beta}"
          f"{'  [DISABLED]' if cycle_beta == 0.0 else ''}")
    print(f"  Train steps/iter = {train_steps}")

    # Track all results
    all_history = []
    best_molecules = []  # (smiles, score) across all iterations
    all_scored_molecules = []  # (smiles, score, route) for top-K results
    total_oracle_calls = 0
    best_reward_so_far = -float("inf")

    # For AUC computation: track top-K averages at each oracle call checkpoint
    auc_oracle_calls = []
    auc_top10 = []
    auc_top30 = []
    auc_top50 = []
    # Internal diversity (1 - mean pairwise Tanimoto) of cumulative top-100 per iter
    int_div_curve = []

    # Replay buffer entries are dicts {"z","reward","fp","smiles"}.
    # Insertion follows a Tanimoto-clustered, novelty-aware policy:
    #   - similar (sim >= threshold) to any entry → replace the lowest-reward
    #     entry in the matched cluster if new reward beats it, else discard
    #   - novel + buffer not full → append unconditionally
    #   - novel + buffer full → replace argmin-reward entry if new reward beats it
    # Training uses a random subset (replay_sample_frac) of the buffer per iter.
    replay_size = ft_cfg.get("replay_size", 200)
    replay_sim_threshold = ft_cfg.get("replay_sim_threshold", 0.85)
    replay_sample_frac = ft_cfg.get("replay_sample_frac", 0.2)
    # Hard gate on replay-buffer admission via cycle error. Default = inf
    # (disabled) because for any AE not trained with explicit cycle-consistency
    # loss, E(D(z)) ≠ z by construction for flow-sampled z (D is many-to-one),
    # and a tight threshold would reject every sample. The cycle signal is
    # instead used as a SOFT down-weight via the beta term in trainer.
    consistency_threshold = float(ft_cfg.get("consistency_threshold", float("inf")))
    replay_z = []
    print(f"  Replay: size={replay_size}, sample_frac={replay_sample_frac}, "
          f"sim_thr={replay_sim_threshold}, consistency_thr={consistency_threshold:g}")

    def _insert_replay(buffer, z, reward, fp, smiles, e_cyc):
        """Insert into buffer using the clustered policy.

        Returns one of: "appended", "replaced", "discarded".
        """
        if buffer:
            sims = DataStructs.BulkTanimotoSimilarity(fp, [e["fp"] for e in buffer])
            similar_idx = [i for i, s in enumerate(sims) if s >= replay_sim_threshold]
        else:
            similar_idx = []
        entry = {"z": z, "reward": reward, "fp": fp, "smiles": smiles, "e_cyc": e_cyc}
        if similar_idx:
            j = min(similar_idx, key=lambda i: buffer[i]["reward"])
            if reward > buffer[j]["reward"]:
                buffer[j] = entry
                return "replaced"
            return "discarded"
        else:
            if len(buffer) < replay_size:
                buffer.append(entry)
                return "appended"
            else:
                k = min(range(len(buffer)), key=lambda i: buffer[i]["reward"])
                if reward > buffer[k]["reward"]:
                    buffer[k] = entry
                    return "replaced"
                return "discarded"

    # Each iter samples a batch of OVERSAMPLE_FACTOR * samples_per_iter
    # latents, decodes all of them, and keeps the first samples_per_iter that
    # produce valid routes. If a single oversampled batch isn't enough (e.g.,
    # validity collapses below ~1/factor), additional 2× batches are drawn up
    # to MAX_OVERSAMPLE_ROUNDS times before giving up.
    OVERSAMPLE_FACTOR = 2
    MAX_OVERSAMPLE_ROUNDS = 5

    for iteration in range(num_iterations):
        # τ annealing: linearly increase from tau_min to tau_max
        trainer.tau = tau_min + (tau_max - tau_min) * iteration / max(num_iterations - 1, 1)

        # 1+2. Sample z₁ (with 2× oversample) and decode. Keep the first
        # samples_per_iter valid routes — guarantees a fixed oracle budget
        # per iter regardless of the AE's per-sample decode validity.
        collected_z = []
        collected_routes = []
        total_attempts = 0
        for round_idx in range(MAX_OVERSAMPLE_ROUNDS):
            batch_size = samples_per_iter * OVERSAMPLE_FACTOR
            z_batch = trainer.sample(batch_size, ode_steps=ode_steps)
            for i in range(batch_size):
                if len(collected_routes) >= samples_per_iter:
                    break
                total_attempts += 1                       # count actual decode attempts
                z_i = z_batch[i:i+1]
                route = ae_model.decode_autoregressive(z_i, compat, bb_mols, greedy=True)
                if route is not None and route.final_product is not None:
                    collected_z.append(z_batch[i])
                    collected_routes.append(route)
            if len(collected_routes) >= samples_per_iter:
                break

        n_valid = len(collected_routes)
        if n_valid < samples_per_iter:
            print(f"  WARN iter {iteration+1}: only collected {n_valid}/"
                  f"{samples_per_iter} valid after {total_attempts} attempts; "
                  f"oracle budget for this iter is short.")
        validity = n_valid / max(total_attempts, 1)

        if n_valid == 0:
            # Skip this iter entirely; nothing to train on.
            continue

        # Stack collected valid z's into a single tensor. All entries valid.
        z1 = torch.stack(collected_z).to(device)
        smiles_list = [r.final_product for r in collected_routes]
        decoded_routes = list(enumerate(collected_routes))

        # 3. Compute cycle errors e_cyc = ||z - E(D(z))||₂ for the n_valid
        # collected routes. All entries are valid by construction.
        z_cyc = encode_routes(ae_model, collected_routes, device)
        e_cyc = (z1 - z_cyc).norm(dim=-1)

        n_pass_gate = int((e_cyc < consistency_threshold).sum().item())
        e_cyc_min = float(e_cyc.min().item())
        e_cyc_med = float(e_cyc.median().item())

        # 4. Evaluate via oracle (n_valid calls).
        rewards = torch.zeros(n_valid, device=device)
        scores = oracle(smiles_list)
        if not isinstance(scores, list):
            scores = [scores]

        n_appended = 0
        n_replaced = 0
        n_discarded = 0
        for idx, score in enumerate(scores):
            rewards[idx] = float(score)
            smi = smiles_list[idx]
            best_molecules.append((smi, float(score)))
            all_scored_molecules.append((smi, float(score), collected_routes[idx]))
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            e_cyc_val = float(e_cyc[idx].item())
            if e_cyc_val >= consistency_threshold:
                continue                              # not cycle-consistent
            outcome = _insert_replay(replay_z, z1[idx].detach().clone(),
                                     float(score), fp, smi, e_cyc_val)
            if outcome == "appended":
                n_appended += 1
            elif outcome == "replaced":
                n_replaced += 1
            else:
                n_discarded += 1
        total_oracle_calls += n_valid

        # 4. Combine current batch with a random subset of the replay buffer.
        if replay_z:
            n_sample = max(1, int(round(replay_sample_frac * len(replay_z))))
            n_sample = min(n_sample, len(replay_z))
            sampled = _random.sample(replay_z, n_sample)
            replay_z_tensor = torch.stack([e["z"] for e in sampled]).to(device)
            replay_r_tensor = torch.tensor([e["reward"] for e in sampled], device=device)
            replay_e_tensor = torch.tensor([e["e_cyc"] for e in sampled], device=device)
            # Add small noise to replay z to prevent overfitting to exact points
            replay_z_noisy = replay_z_tensor + torch.randn_like(replay_z_tensor) * 0.05
            train_z = torch.cat([z1, replay_z_noisy], dim=0)
            train_r = torch.cat([rewards, replay_r_tensor], dim=0)
            train_e = torch.cat([e_cyc, replay_e_tensor], dim=0)
        else:
            train_z = z1
            train_r = rewards
            train_e = e_cyc

        metrics = trainer.train_on_batch(train_z, train_r,
                                         cycle_errors=train_e,
                                         train_steps=train_steps)
        metrics["validity"] = validity
        metrics["n_valid"] = n_valid
        metrics["total_oracle_calls"] = total_oracle_calls

        # Track best
        if rewards.max().item() > best_reward_so_far:
            best_reward_so_far = rewards.max().item()

        all_history.append(metrics)

        # Print progress every iteration
        avg_valid_reward = (rewards.sum() / max(n_valid, 1)).item()
        # Cumulative top-10 average from best_molecules
        best_molecules.sort(key=lambda x: x[1], reverse=True)
        top10_avg = np.mean([s for _, s in best_molecules[:10]]) if len(best_molecules) >= 10 else 0.0

        # Track AUC data points (cumulative top-K average reward, with duplicates)
        all_scored_sorted = sorted(all_scored_molecules, key=lambda x: x[1], reverse=True)
        t10 = np.mean([s for _, s, _ in all_scored_sorted[:10]]) if len(all_scored_sorted) >= 10 else 0.0
        t30 = np.mean([s for _, s, _ in all_scored_sorted[:30]]) if len(all_scored_sorted) >= 30 else 0.0
        t50 = np.mean([s for _, s, _ in all_scored_sorted[:50]]) if len(all_scored_sorted) >= 50 else 0.0
        # Internal diversity over cumulative global top-100 (with duplicates)
        top100_smis = [s for s, _, _ in all_scored_sorted[:100]]
        int_div = internal_diversity(top100_smis) if top100_smis else 0.0
        auc_oracle_calls.append(total_oracle_calls)
        auc_top10.append(t10)
        auc_top30.append(t30)
        auc_top50.append(t50)
        int_div_curve.append(int_div)

        cyc_str = (f"e_cyc[min={e_cyc_min:.1e},med={e_cyc_med:.1e}]  "
                   if np.isfinite(e_cyc_min) else "")
        gate_str = f"pass_gate={n_pass_gate}/{n_valid}  "
        if n_valid > 0 and n_pass_gate == 0:
            gate_str = "⚠ " + gate_str
        print(f"Iter {iteration+1}/{num_iterations}  "
              f"avg_valid={avg_valid_reward:.4f}  "
              f"max={metrics['max_reward']:.4f}  "
              f"top10={top10_avg:.4f}  "
              f"best={best_reward_so_far:.4f}  "
              f"valid={validity*100:.1f}%  "
              f"τ={trainer.tau:.2f}  β={trainer.beta:.2f}  "
              f"fm={metrics['fm_loss']:.4f}  "
              f"w2={metrics['w2_loss']:.4f}  "
              f"{cyc_str}"
              f"{gate_str}"
              f"replay+={n_appended + n_replaced}  replay_sz={len(replay_z)}  "
              f"oracle={total_oracle_calls}  "
              f"int_div={int_div:.4f}")

        # Save checkpoint every 25 iterations
        if (iteration + 1) % 25 == 0 and not args.no_save_checkpoint:
            ckpt_path = os.path.join(ckpt_dir, f"flow_finetune{tag_suffix}_{run_name}_latest.pt")
            trainer.save_checkpoint(ckpt_path, iteration + 1, best_reward_so_far)

    # ---- Save final model and results ----
    if not args.no_save_checkpoint:
        final_ckpt_path = os.path.join(ckpt_dir, f"flow_finetune{tag_suffix}_{run_name}_best.pt")
        trainer.save_checkpoint(final_ckpt_path, num_iterations, best_reward_so_far)
        print(f"\nSaved fine-tuned model to {final_ckpt_path}")
    else:
        print(f"\n[--no_save_checkpoint] skipping checkpoint save")

    # Sort best molecules (KEEP duplicates — global top-100 with multiplicity)
    best_molecules.sort(key=lambda x: x[1], reverse=True)
    top_100 = best_molecules[:100]  # with duplicates

    all_scored_molecules.sort(key=lambda x: x[1], reverse=True)
    # Note: keep duplicates in all_scored_molecules; do NOT dedupe

    # ---- Compute AUC metrics ----
    # AUC = area under top-K curve / budget
    # x-axis: oracle calls (raw), y-axis: top-K avg score (already in [0,1])
    # Divide by budget to normalize area to [0,1]
    oc = np.array(auc_oracle_calls, dtype=np.float64)

    def compute_auc(x, y, budget):
        """Trapezoidal AUC normalized by oracle budget."""
        if len(x) < 2:
            return 0.0
        return float(np.trapz(y, x)) / budget

    auc_10 = compute_auc(oc, np.array(auc_top10), total_oracle_calls)
    auc_30 = compute_auc(oc, np.array(auc_top30), total_oracle_calls)
    auc_50 = compute_auc(oc, np.array(auc_top50), total_oracle_calls)
    final_int_div = int_div_curve[-1] if int_div_curve else 0.0

    # ---- Save results to results/ folder ----
    results_dir = (os.path.join("results", ae_tag, run_name) if ae_tag
                   else os.path.join("results", run_name))
    os.makedirs(results_dir, exist_ok=True)

    # Save GLOBAL top-100 molecules (with duplicates) with SMILES, score, and pathway
    top100_with_routes = all_scored_molecules[:100]
    top100_results = []
    for rank, (smi, score, route) in enumerate(top100_with_routes, 1):
        entry = {
            "rank": rank,
            "smiles": smi,
            "score": score,
            "pathway": route.token_sequence_str() if route else "N/A",
        }
        top100_results.append(entry)

    # Save as pickle (full route objects)
    with open(os.path.join(results_dir, "top100.pkl"), "wb") as f:
        pickle.dump({
            "property": prop_name,
            "top100": [(smi, score, route) for smi, score, route in top100_with_routes],
            "auc10": auc_10,
            "auc30": auc_30,
            "auc50": auc_50,
            "internal_diversity_top100": final_int_div,
            "auc_oracle_calls": auc_oracle_calls,
            "auc_top10_curve": auc_top10,
            "auc_top30_curve": auc_top30,
            "auc_top50_curve": auc_top50,
            "int_div_curve": int_div_curve,
            "total_oracle_calls": total_oracle_calls,
        }, f)

    # Save as human-readable text
    with open(os.path.join(results_dir, "top100.txt"), "w") as f:
        f.write(f"Property: {prop_name}\n")
        f.write(f"Total oracle calls: {total_oracle_calls}\n")
        f.write(f"AUC Top-10: {auc_10:.4f}\n")
        f.write(f"AUC Top-30: {auc_30:.4f}\n")
        f.write(f"AUC Top-50: {auc_50:.4f}\n")
        f.write(f"Top-100 internal diversity (1 - mean Tanimoto): {final_int_div:.4f}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Top-100 molecules (GLOBAL — duplicates kept, sorted by score)\n")
        f.write(f"{'='*80}\n")
        for entry in top100_results:
            f.write(f"\n#{entry['rank']}  score={entry['score']:.4f}\n")
            f.write(f"  SMILES:   {entry['smiles']}\n")
            f.write(f"  Pathway:  {entry['pathway']}\n")
        # Per-iteration AUC + internal diversity curves
        f.write(f"\n{'='*80}\n")
        f.write(f"Per-iteration cumulative top-K AUC curves and internal diversity\n")
        f.write(f"  (top-K averages computed over cumulative top-K with duplicates)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'iter':>5}  {'oracle':>7}  {'top10':>9}  {'top30':>9}  {'top50':>9}  {'int_div':>9}\n")
        for i in range(len(auc_oracle_calls)):
            f.write(f"{i+1:>5}  {auc_oracle_calls[i]:>7}  "
                    f"{auc_top10[i]:>9.4f}  {auc_top30[i]:>9.4f}  {auc_top50[i]:>9.4f}  "
                    f"{int_div_curve[i]:>9.4f}\n")

    # Save training history
    history_path = os.path.join(out_dir, f"flow_finetune{tag_suffix}_{run_name}_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump({
            "property": prop_name,
            "history": all_history,
            "best_molecules": top_100,
            "total_oracle_calls": total_oracle_calls,
            "best_reward": best_reward_so_far,
            "auc10": auc_10,
            "auc30": auc_30,
            "auc50": auc_50,
            "internal_diversity_top100": final_int_div,
            "auc_oracle_calls": auc_oracle_calls,
            "auc_top10_curve": auc_top10,
            "auc_top30_curve": auc_top30,
            "auc_top50_curve": auc_top50,
            "int_div_curve": int_div_curve,
        }, f)
    print(f"Saved training history to {history_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ORW-CFM-W2 Fine-tuning Complete ({prop_name})")
    print(f"{'='*60}")
    print(f"  Total oracle calls:    {total_oracle_calls}")
    print(f"  Best reward:           {best_reward_so_far:.4f}")
    if top_100:
        top10_avg = np.mean([s for _, s in top_100[:10]])
        print(f"  Top-1 reward:          {top_100[0][1]:.4f}")
        print(f"  Top-10 avg reward:     {top10_avg:.4f}")
    print(f"  Unique molecules:      {len(set(s for s, _ in best_molecules))}")
    print(f"  AUC Top-10:            {auc_10:.4f}")
    print(f"  AUC Top-30:            {auc_30:.4f}")
    print(f"  AUC Top-50:            {auc_50:.4f}")
    print(f"  Top-100 internal div:  {final_int_div:.4f}")
    print(f"  Results saved to:      {results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

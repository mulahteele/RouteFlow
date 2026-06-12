"""Phase 3B eval: Evaluate ORW-CFM-W2 fine-tuned flow model.

Plot reward curves, validity curves, W2 distance, and diversity metrics.
"""

import os
import pickle
import argparse

import yaml
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Phase 3B eval: Evaluate fine-tuned flow")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--property", type=str, default="GSK3B",
                        help="Oracle property name")
    args = parser.parse_args()

    prop_name = args.property

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    out_dir = paths["processed_dir"]

    # ---- Load training history ----
    history_path = os.path.join(out_dir, f"flow_finetune_{prop_name}_history.pkl")
    with open(history_path, "rb") as f:
        data = pickle.load(f)

    history = data["history"]
    best_molecules = data["best_molecules"]
    total_oracle_calls = data["total_oracle_calls"]

    # ---- Extract curves ----
    iterations = list(range(1, len(history) + 1))
    mean_rewards = [h["mean_reward"] for h in history]
    max_rewards = [h["max_reward"] for h in history]
    validities = [h["validity"] for h in history]
    fm_losses = [h["fm_loss"] for h in history]
    w2_losses = [h["w2_loss"] for h in history]
    total_losses = [h["total_loss"] for h in history]

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"  ORW-CFM-W2 Fine-tuning Evaluation ({prop_name})")
    print(f"{'='*60}")

    print(f"\n  Reward Curve:")
    print(f"    {'Iteration':<12s} {'Mean':>8s} {'Max':>8s} {'Validity':>10s}")
    print(f"    {'-'*40}")
    for i in [0, 9, 24, 49, 74, 99]:
        if i < len(history):
            print(f"    {i+1:<12d} {mean_rewards[i]:8.4f} {max_rewards[i]:8.4f} "
                  f"{validities[i]*100:9.1f}%")

    print(f"\n  Loss Curve:")
    print(f"    {'Iteration':<12s} {'FM Loss':>10s} {'W2 Loss':>10s} {'Total':>10s}")
    print(f"    {'-'*44}")
    for i in [0, 9, 24, 49, 74, 99]:
        if i < len(history):
            print(f"    {i+1:<12d} {fm_losses[i]:10.4f} {w2_losses[i]:10.4f} "
                  f"{total_losses[i]:10.4f}")

    # ---- Diversity metrics ----
    if best_molecules:
        all_smiles = [s for s, _ in best_molecules]
        unique_smiles = set(all_smiles)

        # Scaffold diversity (top 100)
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            top100_smiles = [s for s, _ in best_molecules[:100]]
            scaffolds = set()
            for smi in top100_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    try:
                        scf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                        scaffolds.add(scf)
                    except Exception:
                        pass
            n_scaffolds = len(scaffolds)
        except ImportError:
            n_scaffolds = -1

        top10_avg = np.mean([s for _, s in best_molecules[:10]]) if len(best_molecules) >= 10 else 0
        top100_avg = np.mean([s for _, s in best_molecules[:100]]) if len(best_molecules) >= 100 else 0

        print(f"\n  Best Molecules:")
        print(f"    Total oracle calls:       {total_oracle_calls}")
        print(f"    Top-1 reward:             {best_molecules[0][1]:.4f}")
        print(f"    Top-10 avg reward:        {top10_avg:.4f}")
        print(f"    Top-100 avg reward:       {top100_avg:.4f}")
        print(f"    Total valid molecules:    {len(all_smiles)}")
        print(f"    Unique molecules:         {len(unique_smiles)}")
        print(f"    Diversity (unique/total):  {len(unique_smiles)/max(len(all_smiles),1)*100:.1f}%")
        if n_scaffolds >= 0:
            print(f"    Unique scaffolds (top100): {n_scaffolds}")

        print(f"\n  Top-5 Molecules:")
        for rank, (smi, score) in enumerate(best_molecules[:5], 1):
            print(f"    #{rank}: score={score:.4f}  SMILES={smi[:80]}")

    print(f"\n{'='*60}")

    # Save summary
    summary_path = os.path.join(out_dir, f"flow_finetune_{prop_name}_eval_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump({
            "mean_rewards": mean_rewards,
            "max_rewards": max_rewards,
            "validities": validities,
            "fm_losses": fm_losses,
            "w2_losses": w2_losses,
            "total_oracle_calls": total_oracle_calls,
        }, f)
    print(f"Saved eval summary to {summary_path}")


if __name__ == "__main__":
    main()

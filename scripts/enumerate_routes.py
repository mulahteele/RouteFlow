"""Phase 1: Enumerate routes with per-rxn-count quotas, dedup-and-split.

Generates two separate dedup'd pools (sharing one global signature set):
  - val/test pool: unfiltered, used to fill validation and test splits
  - train pool:    only routes whose final product satisfies the configured
                   drug-likeness filter (SA<2.5 AND QED>0.55 by default)

Both pools follow the same per-rxn-count ratio. Each is shuffled independently
and sliced into its target split. Validation and test contain no SA/QED
filtering, exactly as before; only the training set is restricted.
"""

import os
import sys
import pickle
import argparse
import random

import yaml
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
from collections import Counter

# Make RDKit Contrib's SA scorer importable
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # noqa: E402

from routeflow.chem.template import read_templates
from routeflow.chem.compatibility import CompatibilityMatrix
from routeflow.data.enumerate_routes import enumerate_routes
from routeflow.data.route import TokenType


# Training-set drug-likeness filter constants.
SA_MAX = 2.5
QED_MIN = 0.5

# Hard test-set constants — drug-like but synthetically challenging.
# Same QED_MIN as Easy (drug-like control), but SA must EXCEED this threshold.
SA_HARD_MIN = 3.5
N_HARD_TEST = 10000


def train_quality_filter(route) -> bool:
    """Return True iff the route's final product passes SA<SA_MAX AND QED>QED_MIN."""
    smi = getattr(route, "final_product", None)
    if not smi:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        if sascorer.calculateScore(mol) >= SA_MAX:
            return False
        if QED.qed(mol) <= QED_MIN:
            return False
    except Exception:
        return False
    return True


def _token_signature(route) -> tuple:
    """Tuple of (token_type, value) per token, dropping START/END."""
    sig = []
    for t in route.tokens:
        if t.token_type in (TokenType.START, TokenType.END):
            continue
        v = t.value if t.value is not None else 0
        sig.append((int(t.token_type), int(v)))
    return tuple(sig)


def generate_dedup_pool(
    compat,
    target_count: int,
    max_heavy_atoms: int,
    seed: int,
    num_workers: int = 32,
    overshoot: float = 1.05,
    max_rounds: int = 40,
    max_reactions: int = 5,
    sa_max: float = None,
    qed_min: float = None,
    sa_min: float = None,
    seen_sigs: set = None,
) -> list:
    """Generate `target_count` unique routes (any rxn count up to max_reactions).

    If sa_max/qed_min are provided, the SA<sa_max AND QED>qed_min filter is
    applied INSIDE each worker (parallel) — no single-thread filter bottleneck
    in the parent. The returned routes are already filtered; the parent only
    runs dedup on token signatures (cheap hash lookups).

    Args:
        seen_sigs: shared dedup set across pools to keep them disjoint.

    Returns a flat list of routes. No specific order — caller shuffles + splits.
    """
    if seen_sigs is None:
        seen_sigs = set()
    routes = []
    round_num = 0
    last_count = -1
    stuck_rounds = 0
    filter_active = (sa_max is not None) or (qed_min is not None) or (sa_min is not None)

    while len(routes) < target_count:
        round_num += 1
        remaining = target_count - len(routes)
        to_gen = max(int(remaining * overshoot), 20000)

        print(f"  round {round_num}: need {remaining} more "
              f"({'filtered' if filter_active else 'raw'}), "
              f"asking workers for {to_gen}...")

        raw = enumerate_routes(
            compat=compat,
            num_routes=to_gen,
            max_heavy_atoms=max_heavy_atoms,
            max_reactions=max_reactions,
            seed=seed + round_num * 1000,
            num_workers=num_workers,
            sa_max=sa_max,
            qed_min=qed_min,
            sa_min=sa_min,
        )

        n_dups = 0
        for r in raw:
            sig = _token_signature(r)
            if sig in seen_sigs:
                n_dups += 1
                continue
            seen_sigs.add(sig)
            routes.append(r)
            if len(routes) >= target_count:
                break

        print(f"    -> {len(routes)}/{target_count} kept "
              f"({len(raw)} returned by workers, {n_dups} dups against seen_sigs)")

        # Stuck detection
        if len(routes) == last_count:
            stuck_rounds += 1
            if stuck_rounds >= 3 or round_num >= max_rounds:
                print(f"    Warning: stuck at {len(routes)}/{target_count}, moving on")
                break
        else:
            stuck_rounds = 0
            last_count = len(routes)

    return routes[:target_count]


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Enumerate routes")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--val_test_only_filtered", action="store_true",
        help="Skip train pool generation. Reuse the existing routes_train.pkl "
             "(its token signatures populate the dedup set), then regenerate "
             "val and test pools WITH the SA/QED filter applied. Overwrites "
             "the existing routes_val.pkl and routes_test_easy.pkl in-place.",
    )
    parser.add_argument(
        "--hard_test_only", action="store_true",
        help=f"Generate a HARD test set: SA > {SA_HARD_MIN} AND QED > {QED_MIN} "
             f"(drug-like but synthetically challenging — OOD on synthesis axis). "
             f"Loads existing train/val/test signatures to ensure disjoint set. "
             f"Saves to data/processed/routes_test_hard.pkl; does NOT modify any "
             f"existing routes_*.pkl. Target size = {N_HARD_TEST}.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    out_dir = paths["processed_dir"]

    compat_path = os.path.join(out_dir, "compatibility_matrix.pkl")
    print(f"Loading compatibility matrix from {compat_path}...")
    with open(compat_path, "rb") as f:
        compat_data = pickle.load(f)

    bb_smiles = compat_data["bb_smiles"]
    bb_mols = [Chem.MolFromSmiles(s) for s in bb_smiles]
    reactions = compat_data["reactions"]
    compat = CompatibilityMatrix(
        bb_smiles, bb_mols, reactions, matrix=compat_data["matrix"]
    )
    print(f"  {compat.num_bb} BBs, {compat.num_rxn} reactions")

    n_train = int(data_cfg["num_train_routes"])
    n_val = int(data_cfg["num_val_routes"])
    n_test = int(data_cfg["num_test_routes"])

    max_heavy_atoms = data_cfg.get("max_heavy_atoms", 80)
    seed = data_cfg.get("seed", 42)
    num_workers = data_cfg.get("num_workers", 32)
    max_reactions = int(data_cfg.get("max_reactions", 5))

    val_test_total = n_val + n_test

    # ------------------------------------------------------------------
    # Mode C (--hard_test_only): generate ONE-OFF hard test set
    # (SA > SA_HARD_MIN AND QED > QED_MIN) disjoint from existing splits.
    # Does NOT touch routes_{train,val,test}.pkl.
    # ------------------------------------------------------------------
    if args.hard_test_only:
        print(f"\n[hard_test_only mode]")
        print(f"  Target: {N_HARD_TEST} routes with SA > {SA_HARD_MIN} AND QED > {QED_MIN}")
        print(f"  Output: data/processed/routes_test_hard.pkl")
        print(f"  Dedup: against existing train + val + test_easy")

        seen_sigs: set = set()
        for split in ("train", "val", "test_easy"):
            p = os.path.join(out_dir, f"routes_{split}.pkl")
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"--hard_test_only requires {p} to exist for dedup."
                )
            with open(p, "rb") as f:
                rs = pickle.load(f)["routes"]
            for r in rs:
                seen_sigs.add(_token_signature(r))
            print(f"  loaded {len(rs):,} {split} routes (seen_sigs now {len(seen_sigs):,})")

        print(f"\nGenerating hard test pool (worker-side filter)...")
        hard_pool = generate_dedup_pool(
            compat=compat,
            target_count=N_HARD_TEST,
            max_heavy_atoms=max_heavy_atoms,
            seed=seed + 7_777_777,
            num_workers=num_workers,
            overshoot=1.1,
            max_rounds=40,
            max_reactions=max_reactions,
            sa_min=SA_HARD_MIN,
            qed_min=QED_MIN,
            seen_sigs=seen_sigs,
        )
        print(f"  Generated hard pool: {len(hard_pool)} unique filtered routes")

        rng = random.Random(seed + 7_777_777)
        rng.shuffle(hard_pool)
        hard_routes = hard_pool[:N_HARD_TEST]

        n_rxns = [r.num_reactions for r in hard_routes]
        n_tokens = [r.num_tokens for r in hard_routes]
        rxn_dist = Counter(n_rxns)
        print(f"\n[test_hard] {len(hard_routes)} routes")
        print(f"  Reactions: min={min(n_rxns)} max={max(n_rxns)} avg={np.mean(n_rxns):.2f}")
        print(f"  Distribution: {dict(sorted(rxn_dist.items()))}")
        print(f"  Tokens:    min={min(n_tokens)} max={max(n_tokens)} avg={np.mean(n_tokens):.1f}")

        save_path = os.path.join(out_dir, "routes_test_hard.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"routes": hard_routes}, f)
        print(f"  Saved to {save_path}")
        print("\nHard test set generation complete.")
        return

    # ------------------------------------------------------------------
    # Mode A (--val_test_only_filtered): reuse existing train, regen val/test WITH filter.
    # ------------------------------------------------------------------
    if args.val_test_only_filtered:
        train_path = os.path.join(out_dir, "routes_train.pkl")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"--val_test_only_filtered requires {train_path} to exist."
            )

        print(f"\n[val_test_only_filtered mode]")
        print(f"  train: REUSE {train_path} (unchanged)")
        print(f"  val:   {n_val}   (filtered: SA<{SA_MAX} AND QED>{QED_MIN})")
        print(f"  test:  {n_test}  (filtered: SA<{SA_MAX} AND QED>{QED_MIN})")
        print(f"  max_heavy_atoms: {max_heavy_atoms}  max_reactions: {max_reactions}")

        print(f"\nLoading existing train pool to populate seen_sigs (dedup)...")
        with open(train_path, "rb") as f:
            existing_train = pickle.load(f)["routes"]
        print(f"  loaded {len(existing_train)} train routes")
        seen_sigs: set = set()
        for r in existing_train:
            seen_sigs.add(_token_signature(r))
        print(f"  seen_sigs initialized with {len(seen_sigs)} signatures from train")
        del existing_train  # free memory

        print("\nGenerating val/test pool (SA<{:.2f} AND QED>{:.2f}, worker-side filter)...".format(SA_MAX, QED_MIN))
        val_test_pool = generate_dedup_pool(
            compat=compat,
            target_count=val_test_total,
            max_heavy_atoms=max_heavy_atoms,
            seed=seed,
            num_workers=num_workers,
            overshoot=1.05,
            max_rounds=40,
            max_reactions=max_reactions,
            sa_max=SA_MAX,
            qed_min=QED_MIN,
            seen_sigs=seen_sigs,
        )
        print(f"  Generated val/test pool: {len(val_test_pool)} unique filtered routes")

        rng = random.Random(seed)
        rng.shuffle(val_test_pool)

        splits = {
            "val":       val_test_pool[:n_val],
            "test_easy": val_test_pool[n_val:n_val + n_test],
        }
    # ------------------------------------------------------------------
    # Mode B (default): full enumeration (train + val/test).
    # ------------------------------------------------------------------
    else:
        print(f"\nPool plan (NO per-rxn-count quotas — natural distribution):")
        print(f"  train: {n_train}  (filtered: SA<{SA_MAX} AND QED>{QED_MIN})")
        print(f"  val:   {n_val}    (no filter)")
        print(f"  test:  {n_test}   (no filter)")
        print(f"  max_heavy_atoms: {max_heavy_atoms}  max_reactions: {max_reactions}")

        # Shared dedup signature set across both pools so train ∩ (val ∪ test) = ∅.
        seen_sigs = set()

        # 1) Generate val/test pool first (no filter).
        print("\nGenerating val/test pool (no filter)...")
        val_test_pool = generate_dedup_pool(
            compat=compat,
            target_count=val_test_total,
            max_heavy_atoms=max_heavy_atoms,
            seed=seed,
            num_workers=num_workers,
            max_reactions=max_reactions,
            seen_sigs=seen_sigs,
        )
        print(f"  Generated val/test pool: {len(val_test_pool)} unique routes")

        # 2) Generate train pool with SA/QED filter applied INSIDE workers (parallel).
        # overshoot=1.05: workers handle filter loss internally; main only needs to
        # cover the small dedup loss against seen_sigs (~few %).
        print("\nGenerating train pool (SA<{:.2f} AND QED>{:.2f}, worker-side filter)...".format(SA_MAX, QED_MIN))
        train_pool = generate_dedup_pool(
            compat=compat,
            target_count=n_train,
            max_heavy_atoms=max_heavy_atoms,
            seed=seed + 999_999,
            num_workers=num_workers,
            overshoot=1.05,
            max_rounds=40,
            max_reactions=max_reactions,
            sa_max=SA_MAX,
            qed_min=QED_MIN,
            seen_sigs=seen_sigs,
        )
        print(f"  Generated train pool: {len(train_pool)} unique filtered routes")

        rng = random.Random(seed)
        rng.shuffle(train_pool)
        rng.shuffle(val_test_pool)

        splits = {
            "train":     train_pool[:n_train],
            "val":       val_test_pool[:n_val],
            "test_easy": val_test_pool[n_val:n_val + n_test],
        }

    # ------------------------------------------------------------------
    # Save splits (same code path for both modes; train only written in default mode).
    # ------------------------------------------------------------------
    for split, routes in splits.items():
        n_rxns = [r.num_reactions for r in routes]
        n_tokens = [r.num_tokens for r in routes]
        rxn_dist = Counter(n_rxns)
        print(f"\n[{split}] {len(routes)} routes")
        print(f"  Reactions: min={min(n_rxns)} max={max(n_rxns)} avg={np.mean(n_rxns):.2f}")
        print(f"  Distribution: {dict(sorted(rxn_dist.items()))}")
        print(f"  Tokens:    min={min(n_tokens)} max={max(n_tokens)} avg={np.mean(n_tokens):.1f}")

        save_path = os.path.join(out_dir, f"routes_{split}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"routes": routes}, f)
        print(f"  Saved to {save_path}")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()

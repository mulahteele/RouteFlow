"""Phase 0: Build BB × Template × Position compatibility matrix."""

import os
import pickle
import argparse

import numpy as np
import yaml

from routeflow.chem.template import read_templates
from routeflow.chem.building_block import load_building_blocks, compute_morgan_fingerprints
from routeflow.chem.compatibility import CompatibilityMatrix


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Build compatibility matrix")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    precomp = cfg["precompute"]
    out_dir = paths["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load templates
    print("Loading reaction templates...")
    reactions = read_templates(paths["template_file"])
    print(f"  Loaded {len(reactions)} reaction templates")
    for rxn in reactions[:5]:
        print(f"    {rxn}")

    # Load building blocks
    print("Loading building blocks...")
    bb_smiles, bb_mols = load_building_blocks(paths["bb_file"])
    print(f"  Loaded {len(bb_smiles)} unique building blocks")

    # Compute and save Morgan fingerprints
    print("Computing Morgan fingerprints...")
    bb_fps = compute_morgan_fingerprints(
        bb_mols,
        radius=precomp["fp_radius"],
        nbits=precomp["fp_nbits"],
    )
    fp_path = os.path.join(out_dir, "bb_fingerprints.npy")
    np.save(fp_path, bb_fps)
    print(f"  Saved fingerprints to {fp_path}: shape={bb_fps.shape}")

    # Build compatibility matrix
    print("Building compatibility matrix...")
    compat = CompatibilityMatrix(bb_smiles, bb_mols, reactions)
    print(f"  Matrix shape: {compat.matrix.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(compat.matrix)}")

    # Seed reactions
    seed_rxns = compat.get_seed_reactions()
    print(f"  Seed reactions (all positions fillable): {len(seed_rxns)}")

    # Save
    compat_path = os.path.join(out_dir, "compatibility_matrix.pkl")
    with open(compat_path, "wb") as f:
        pickle.dump({
            "bb_smiles": bb_smiles,
            "reactions": reactions,
            "matrix": compat.matrix,
        }, f)
    print(f"  Saved compatibility matrix to {compat_path}")

    # Save bb_smiles separately for quick access
    smiles_path = os.path.join(out_dir, "bb_smiles.pkl")
    with open(smiles_path, "wb") as f:
        pickle.dump(bb_smiles, f)
    print(f"  Saved BB SMILES to {smiles_path}")

    print("Phase 0 complete.")


if __name__ == "__main__":
    main()

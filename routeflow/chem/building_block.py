"""Building block loading and Morgan fingerprint computation."""

from __future__ import annotations

import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def load_building_blocks(csv_path: str) -> tuple[list[str], list[Chem.Mol]]:
    """Load building blocks from CSV. Returns (smiles_list, rdmol_list).

    Skips invalid molecules. Indices in returned lists are contiguous (0-based BB IDs).
    """
    smiles_list = []
    mol_list = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading building blocks"):
            smi = row["SMILES"].strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                smiles_list.append(canon)
                mol_list.append(mol)
    # Deduplicate by canonical SMILES
    seen = {}
    unique_smiles = []
    unique_mols = []
    for smi, mol in zip(smiles_list, mol_list):
        if smi not in seen:
            seen[smi] = True
            unique_smiles.append(smi)
            unique_mols.append(mol)
    return unique_smiles, unique_mols


def compute_morgan_fingerprints(
    mol_list: list[Chem.Mol],
    radius: int = 2,
    nbits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints for all molecules. Returns (N, nbits) float32 array."""
    fps = np.zeros((len(mol_list), nbits), dtype=np.float32)
    for i, mol in enumerate(tqdm(mol_list, desc="Computing Morgan FPs")):
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr)
        fps[i] = arr
    return fps

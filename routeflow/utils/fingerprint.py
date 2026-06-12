"""Morgan fingerprint utilities."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def compute_mol_fingerprint(
    mol: Chem.Mol | str,
    radius: int = 2,
    nbits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprint for a single molecule.

    Args:
        mol: RDKit Mol or SMILES string
        radius: Morgan FP radius
        nbits: number of bits

    Returns: (nbits,) float32 array
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)

    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def compute_mol_fingerprint_batch(
    smiles_list: list[str],
    radius: int = 2,
    nbits: int = 2048,
) -> np.ndarray:
    """Compute Morgan FPs for a batch of SMILES. Returns (N, nbits) array."""
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        fps[i] = compute_mol_fingerprint(smi, radius, nbits)
    return fps

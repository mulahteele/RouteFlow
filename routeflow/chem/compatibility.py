"""BB × Template × Position compatibility matrix with bitmask encoding."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from .template import Reaction


class CompatibilityMatrix:
    """Compatibility matrix: matrix[bb_idx, rxn_idx] = bitmask of matched positions.

    Bitmask encoding:
        bit 0 (0x01): BB matches reactant template at position 0
        bit 1 (0x02): BB matches reactant template at position 1
        bit 2 (0x04): BB matches reactant template at position 2

    Example for bi-molecular reaction:
        0b01 = matches pos 0 only
        0b10 = matches pos 1 only
        0b11 = matches both pos 0 and pos 1
    """

    def __init__(
        self,
        bb_smiles: list[str],
        bb_mols: list[Chem.Mol],
        reactions: list[Reaction],
        matrix: np.ndarray | None = None,
    ):
        self._bb_smiles = bb_smiles
        self._bb_mols = bb_mols
        self._reactions = reactions
        self._compat_cache: dict[tuple[int, int], np.ndarray] = {}
        if matrix is not None:
            self._matrix = matrix
        else:
            self._matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        """Build the compatibility matrix. Shape: (num_bb, num_rxn), dtype uint8."""
        n_bb = len(self._bb_mols)
        n_rxn = len(self._reactions)
        matrix = np.zeros((n_bb, n_rxn), dtype=np.uint8)

        for j, rxn in enumerate(tqdm(self._reactions, desc="Building compatibility matrix")):
            templates = rxn.reactant_templates
            for i, mol in enumerate(self._bb_mols):
                flag = 0
                for t, tmpl in enumerate(templates):
                    if tmpl.match(mol):
                        flag |= (1 << t)
                matrix[i, j] = flag

        return matrix

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def reactions(self) -> list[Reaction]:
        return self._reactions

    @property
    def bb_smiles(self) -> list[str]:
        return self._bb_smiles

    @property
    def bb_mols(self) -> list[Chem.Mol]:
        return self._bb_mols

    @property
    def num_bb(self) -> int:
        return len(self._bb_smiles)

    @property
    def num_rxn(self) -> int:
        return len(self._reactions)

    def get_compatible_bbs(self, rxn_idx: int, position: int) -> np.ndarray:
        """Return array of BB indices compatible with (rxn_idx, position)."""
        key = (rxn_idx, position)
        if key in self._compat_cache:
            return self._compat_cache[key]
        bit = 1 << position
        col = self._matrix[:, rxn_idx]
        result = np.where(np.bitwise_and(col, bit) > 0)[0]
        self._compat_cache[key] = result
        return result

    def get_seed_reactions(self) -> np.ndarray:
        """Return indices of reactions where all reactant positions have ≥1 compatible BB.

        A 'seed' reaction can start a new branch from scratch (all reactants are BBs).
        """
        seed_indices = []
        for j, rxn in enumerate(self._reactions):
            n = rxn.num_reactants
            all_filled = True
            for pos in range(n):
                if len(self.get_compatible_bbs(j, pos)) == 0:
                    all_filled = False
                    break
            if all_filled:
                seed_indices.append(j)
        return np.array(seed_indices, dtype=np.int64)

    def get_bu_rxn_mask(self, mol_rd: Chem.Mol) -> list[int]:
        """Return reaction indices where mol matches reactant slot 0.

        Stack-top is always routed into slot 0; reactions where mol can only
        match slot 1+ are excluded so the token stream uniquely determines the
        slot assignment (and therefore the product).
        """
        valid = []
        for j, rxn in enumerate(self._reactions):
            if rxn.reactant_templates[0].match(mol_rd):
                valid.append(j)
        return valid

    def get_bu_rxn_mask_with_positions(self, mol_rd: Chem.Mol) -> dict[int, tuple[int, ...]]:
        """Return {rxn_idx: matched_positions} restricted to reactions whose slot 0 mol matches.

        matched_positions is still the full set of slots mol can fill (used by
        enumerate to validate POP compatibility on other slots), but we only
        include reactions where slot 0 is among them.
        """
        result = {}
        for j, rxn in enumerate(self._reactions):
            matched = rxn.match_reactant_templates(mol_rd)
            if 0 in matched:
                result[j] = matched
        return result

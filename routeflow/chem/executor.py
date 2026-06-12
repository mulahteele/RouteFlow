"""RDKit reaction execution and synthesis stack operations."""

from __future__ import annotations

import random
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from .template import Reaction


class ReactionExecutor:
    """Execute a reaction given reactant molecules, returning the product."""

    @staticmethod
    def execute(
        reaction: Reaction,
        reactants: list[Chem.Mol],
    ) -> Optional[Chem.Mol]:
        """Execute reaction on given reactants. Returns first valid product or None."""
        products = reaction.run(reactants)
        if not products:
            return None
        # Take the largest fragment if product contains '.'
        best = products[0]
        best_smi = Chem.MolToSmiles(best)
        if "." in best_smi:
            frags = best_smi.split(".")
            best_smi = max(frags, key=len)
            best = Chem.MolFromSmiles(best_smi)
            if best is None:
                return None
        return best

    @staticmethod
    def num_heavy_atoms(mol: Chem.Mol) -> int:
        """Return number of heavy atoms in molecule."""
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()


class SynthesisStack:
    """Stack data structure for synthesis route construction.

    Maintains a stack of intermediate molecules. Two actions build the tree:
    - Branch: all reactants are BBs, push product
    - React: pop stack top as pos0, remaining positions are BBs or Pop, push product
    """

    def __init__(self):
        self._stack: list[Chem.Mol] = []  # molecule stack
        self._stack_smiles: list[str] = []  # parallel SMILES for tracking

    @property
    def depth(self) -> int:
        return len(self._stack)

    @property
    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def top(self) -> Optional[Chem.Mol]:
        if self._stack:
            return self._stack[-1]
        return None

    def top_smiles(self) -> Optional[str]:
        if self._stack_smiles:
            return self._stack_smiles[-1]
        return None

    def push(self, mol: Chem.Mol, smiles: str):
        self._stack.append(mol)
        self._stack_smiles.append(smiles)

    def pop(self) -> tuple[Chem.Mol, str]:
        mol = self._stack.pop()
        smi = self._stack_smiles.pop()
        return mol, smi

    def peek(self, idx: int = -1) -> Optional[Chem.Mol]:
        """Peek at stack element by index (default: top)."""
        if abs(idx) <= len(self._stack):
            return self._stack[idx]
        return None

    def peek_smiles(self, idx: int = -1) -> Optional[str]:
        if abs(idx) <= len(self._stack_smiles):
            return self._stack_smiles[idx]
        return None

    def can_pop_for_react(self, reaction: Reaction, position: int) -> bool:
        """Check if the next stack molecule matches the reaction's template at position.

        Used for Pop mask validation.
        """
        if self.is_empty:
            return False
        mol = self._stack[-1]
        templates = reaction.reactant_templates
        if position >= len(templates):
            return False
        return templates[position].match(mol)

    def copy(self) -> "SynthesisStack":
        """Create a shallow copy of this stack."""
        new = SynthesisStack()
        new._stack = list(self._stack)
        new._stack_smiles = list(self._stack_smiles)
        return new


def execute_reaction(
    reaction: Reaction,
    reactants: list[Chem.Mol],
) -> Optional[Chem.Mol]:
    """Convenience function: execute reaction, return product mol or None."""
    return ReactionExecutor.execute(reaction, reactants)

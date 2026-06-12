"""Reaction template parsing, reactant counting, and substructure matching."""

from __future__ import annotations

from functools import cached_property
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions


class Template:
    """A molecular substructure pattern (SMARTS)."""

    def __init__(self, smarts: str):
        self._smarts = smarts.strip()

    @property
    def smarts(self) -> str:
        return self._smarts

    @cached_property
    def _rdmol(self) -> Chem.Mol | None:
        return Chem.MolFromSmarts(self._smarts)

    def match(self, mol_rd: Chem.Mol) -> bool:
        if self._rdmol is None or mol_rd is None:
            return False
        return mol_rd.HasSubstructMatch(self._rdmol)

    def __hash__(self):
        return hash(self._smarts)

    def __eq__(self, other):
        return isinstance(other, Template) and self._smarts == other._smarts

    def __repr__(self):
        return f"Template({self._smarts[:40]}...)"

    def __getstate__(self):
        return {"smarts": self._smarts}

    def __setstate__(self, state):
        self._smarts = state["smarts"]


class Reaction:
    """A chemical reaction template from SMARTS."""

    def __init__(self, smarts: str, index: int):
        self._smarts = smarts.strip()
        self._index = index

    @property
    def smarts(self) -> str:
        return self._smarts

    @property
    def index(self) -> int:
        return self._index

    @cached_property
    def _reaction(self) -> rdChemReactions.ChemicalReaction:
        rxn = AllChem.ReactionFromSmarts(self._smarts)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    @cached_property
    def num_reactants(self) -> int:
        return self._reaction.GetNumReactantTemplates()

    @cached_property
    def reactant_templates(self) -> tuple[Template, ...]:
        templates = []
        for i in range(self.num_reactants):
            tmpl = self._reaction.GetReactantTemplate(i)
            smarts = Chem.MolToSmarts(tmpl)
            templates.append(Template(smarts))
        return tuple(templates)

    def match_reactant_templates(self, mol_rd: Chem.Mol) -> tuple[int, ...]:
        """Return indices of reactant template slots this molecule matches."""
        matched = []
        for i, tmpl in enumerate(self.reactant_templates):
            if tmpl.match(mol_rd):
                matched.append(i)
        return tuple(matched)

    def run(self, reactants: Sequence[Chem.Mol]) -> list[Chem.Mol]:
        """Execute reaction on reactants, return list of valid product mols."""
        try:
            product_sets = self._reaction.RunReactants(list(reactants))
        except Exception:
            return []
        products = []
        for ps in product_sets:
            for p in ps:
                try:
                    Chem.SanitizeMol(p)
                    smi = Chem.MolToSmiles(p)
                    if smi:
                        products.append(p)
                except Exception:
                    continue
        return products

    def __repr__(self):
        return f"Reaction(idx={self._index}, n_reactants={self.num_reactants})"

    def __getstate__(self):
        return {"smarts": self._smarts, "index": self._index}

    def __setstate__(self, state):
        self._smarts = state["smarts"]
        self._index = state["index"]


def read_templates(path: str) -> list[Reaction]:
    """Read reaction SMARTS file, one per line. Returns list of Reaction objects."""
    reactions = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                reactions.append(Reaction(line, index=i))
    return reactions

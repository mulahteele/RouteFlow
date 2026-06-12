"""Route representation: token sequence + intermediate molecules + stack trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class TokenType(IntEnum):
    """Token types in route sequence."""
    START = 0
    END = 1
    BRANCH = 2     # Action: Branch
    REACT = 3      # Action: React
    POP = 4        # Source: Pop from stack
    RXN = 5        # Reaction template ID (offset by rxn_id)
    BB = 6         # Building block ID (offset by bb_id)


@dataclass
class RouteToken:
    """A single token in the route sequence."""
    token_type: TokenType
    value: int = 0  # rxn_id for RXN, bb_id for BB, 0 for special tokens

    def __repr__(self):
        if self.token_type == TokenType.START:
            return "[START]"
        elif self.token_type == TokenType.END:
            return "[END]"
        elif self.token_type == TokenType.BRANCH:
            return "[BR]"
        elif self.token_type == TokenType.REACT:
            return "[RE]"
        elif self.token_type == TokenType.POP:
            return "[POP]"
        elif self.token_type == TokenType.RXN:
            return f"r{self.value}"
        elif self.token_type == TokenType.BB:
            return f"b{self.value}"
        return f"?{self.token_type}:{self.value}"


@dataclass
class RouteStep:
    """A single step in the synthesis route."""
    action: str  # "branch" or "react"
    rxn_idx: int
    sources: list[dict]  # each dict: {"type": "bb"|"pop", "bb_idx": int|None}
    product_smiles: str
    product_mol: object = None  # Chem.Mol (not serialized)


@dataclass
class Route:
    """A complete synthesis route.

    Contains:
    - tokens: the flat token sequence for encoder input
    - steps: structured step-by-step info
    - intermediates: SMILES of all intermediate molecules
    - final_product: SMILES of the final product
    """
    tokens: list[RouteToken] = field(default_factory=list)
    steps: list[RouteStep] = field(default_factory=list)
    intermediates: list[str] = field(default_factory=list)
    final_product: Optional[str] = None
    num_reactions: int = 0
    tree_depth: int = 0

    def token_sequence_str(self) -> str:
        return " ".join(str(t) for t in self.tokens)

    def token_types(self) -> list[int]:
        """Return list of token type IDs."""
        return [t.token_type.value for t in self.tokens]

    def token_values(self) -> list[int]:
        """Return list of token values."""
        return [t.value for t in self.tokens]

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def get_bb_ids_used(self) -> list[int]:
        """Return list of BB IDs used in this route."""
        return [t.value for t in self.tokens if t.token_type == TokenType.BB]

    def get_rxn_ids_used(self) -> list[int]:
        """Return list of reaction IDs used in this route."""
        return [t.value for t in self.tokens if t.token_type == TokenType.RXN]

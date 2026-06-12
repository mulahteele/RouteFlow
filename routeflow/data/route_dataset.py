"""PyTorch Dataset for route token sequences."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from routeflow.data.route import Route, TokenType


class RouteDataset(Dataset):
    """Dataset that converts Routes into tensors for autoencoder training.

    Each sample returns:
        token_types: (seq_len,) int tensor of TokenType IDs
        token_values: (seq_len,) int tensor of values (rxn_id, bb_id, etc.)
        seq_len: int, actual sequence length (before padding)
        source_candidates: list of dicts for each source (BB/POP) position
            Each dict: {"candidates": np.array of BB indices, "is_pop_valid": bool}
    """

    def __init__(
        self,
        routes: list[Route],
        bb_fingerprints: np.ndarray,  # (num_bb, fp_nbits)
        compat_matrix=None,  # CompatibilityMatrix, for source candidate masks
    ):
        self.routes = routes
        self.bb_fps = bb_fingerprints
        self.compat = compat_matrix

    def __len__(self) -> int:
        return len(self.routes)

    def __getitem__(self, idx: int) -> dict:
        route = self.routes[idx]

        token_types = torch.tensor(route.token_types(), dtype=torch.long)
        token_values = torch.tensor(route.token_values(), dtype=torch.long)
        seq_len = route.num_tokens

        # Precompute source candidates for each BB/POP position in the target
        # Target is tokens[1:], so source positions are where target is BB or POP
        source_candidates = self._build_source_candidates(route)

        return {
            "token_types": token_types,
            "token_values": token_values,
            "seq_len": seq_len,
            "route_idx": idx,
            "source_candidates": source_candidates,
        }

    def _build_source_candidates(self, route: Route) -> list[dict]:
        """Build per-source-position candidate lists.

        Walk the token sequence to track which rxn_idx and position each
        BB/POP token corresponds to, then look up compat mask.
        """
        if self.compat is None:
            return []

        tokens = route.tokens
        candidates_list = []

        # State tracking
        current_rxn_idx = None
        current_action = None
        current_n_reactants = 0
        source_pos_counter = 0  # which position within current reaction

        for i in range(1, len(tokens)):  # skip START
            t = tokens[i]

            if t.token_type == TokenType.BRANCH:
                current_action = "branch"
                source_pos_counter = 0
            elif t.token_type == TokenType.REACT:
                current_action = "react"
                source_pos_counter = 0
            elif t.token_type == TokenType.RXN:
                current_rxn_idx = t.value
                rxn = self.compat.reactions[current_rxn_idx]
                current_n_reactants = rxn.num_reactants
                if current_action == "branch":
                    source_pos_counter = 0  # all positions are sources
                else:
                    source_pos_counter = 0  # remaining positions (pos0 is implicit pop)
            elif t.token_type in (TokenType.BB, TokenType.POP):
                # Determine which reactant position this corresponds to
                if current_rxn_idx is not None:
                    if current_action == "branch":
                        pos = source_pos_counter
                    else:
                        # React: pos0 is stack top (implicit), sources start at pos1
                        # Find pos0 from matched templates — approximate as pos=0
                        pos = source_pos_counter + 1
                        if pos >= current_n_reactants:
                            pos = min(pos, current_n_reactants - 1)

                    cand_bbs = self.compat.get_compatible_bbs(current_rxn_idx, pos)
                    # is_pop_valid: only for React action, not for Branch
                    is_pop_valid = (current_action == "react")

                    candidates_list.append({
                        "candidates": cand_bbs,
                        "is_pop_valid": is_pop_valid,
                    })
                    source_pos_counter += 1
                else:
                    # Fallback
                    candidates_list.append({
                        "candidates": np.array([], dtype=np.int64),
                        "is_pop_valid": False,
                    })
            # END, START: skip

        return candidates_list

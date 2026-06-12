"""Batch collation with padding and mask construction."""

from __future__ import annotations

import torch


def collate_routes(batch: list[dict]) -> dict:
    """Collate a batch of route samples with padding.

    Returns:
        token_types: (B, max_len) padded token type IDs. Pad value = -1
        token_values: (B, max_len) padded token values. Pad value = 0
        seq_lens: (B,) actual sequence lengths
        padding_mask: (B, max_len) True where padded
        route_indices: (B,) original route indices
        source_candidates: list[list[dict]] — per-batch-element, per-source-position
    """
    batch_size = len(batch)
    max_len = max(b["seq_len"] for b in batch)

    token_types = torch.full((batch_size, max_len), -1, dtype=torch.long)
    token_values = torch.zeros((batch_size, max_len), dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    route_indices = torch.zeros(batch_size, dtype=torch.long)
    source_candidates = []

    for i, b in enumerate(batch):
        sl = b["seq_len"]
        token_types[i, :sl] = b["token_types"]
        token_values[i, :sl] = b["token_values"]
        seq_lens[i] = sl
        padding_mask[i, :sl] = False
        route_indices[i] = b["route_idx"]
        source_candidates.append(b.get("source_candidates", []))

    return {
        "token_types": token_types,
        "token_values": token_values,
        "seq_lens": seq_lens,
        "padding_mask": padding_mask,
        "route_indices": route_indices,
        "source_candidates": source_candidates,  # list, not tensor
    }

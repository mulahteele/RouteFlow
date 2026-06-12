"""Phase 4: ODE integration + decode + oracle evaluation."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from routeflow.models.velocity_net import VelocityNet


def ode_integrate(
    model: VelocityNet,
    z_start: torch.Tensor,       # (B, D)
    num_steps: int = 500,
    endpoint: float = 1.0,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Euler ODE integration from z_start following learned velocity field.

    z_{k+1} = z_k + v_θ(z_k, t_k) * dt

    Returns list of z at each integration step: [z_0, z_1, ..., z_{num_steps}]
    """
    model.eval()
    dt = endpoint / num_steps
    z = z_start.clone().to(device)

    trajectory = [z.cpu().clone()]

    with torch.no_grad():
        for step in range(num_steps):
            t = torch.full((z.shape[0], 1), step * dt, device=device)
            v = model(z, t)
            z = z + v * dt
            trajectory.append(z.cpu().clone())

    return trajectory


def run_inference(
    flow_model: VelocityNet,
    starting_points: np.ndarray,    # (num_starts, D) top-K latent codes
    decode_fn,                       # callable: z (numpy) → list[str] SMILES
    oracle_fn,                       # callable: list[str] → list[float] scores
    num_rounds: int = 10,
    ode_steps: int = 500,
    ode_endpoint: float = 1.0,
    device: str = "cuda",
) -> dict:
    """Run multi-round ODE inference.

    For each starting point:
      - Run 10 rounds of ODE (t: 0→1)
      - Decode z at each round → molecule
      - Evaluate via oracle
      - Keep best molecule across rounds

    Returns:
        results: list of dicts with best molecule per trajectory
        all_scores: (num_starts, num_rounds) all scores
        total_oracle_calls: int
    """
    num_starts = len(starting_points)
    z_current = torch.from_numpy(starting_points).float()

    all_scores = np.zeros((num_starts, num_rounds))
    all_smiles = [[None] * num_rounds for _ in range(num_starts)]
    total_oracle_calls = 0

    for round_idx in tqdm(range(num_rounds), desc="Inference rounds"):
        # ODE integration: z_current → z_next
        trajectory = ode_integrate(
            flow_model, z_current,
            num_steps=ode_steps, endpoint=ode_endpoint, device=device,
        )
        z_next = trajectory[-1]  # (num_starts, D)

        # Decode z to molecules
        z_np = z_next.numpy()
        smiles_list = decode_fn(z_np)

        # Filter valid SMILES and evaluate
        valid_smiles = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            if smi is not None and len(smi) > 0:
                valid_smiles.append(smi)
                valid_indices.append(i)

        if valid_smiles:
            scores = oracle_fn(valid_smiles)
            total_oracle_calls += len(valid_smiles)
            for idx, score in zip(valid_indices, scores):
                all_scores[idx, round_idx] = score
                all_smiles[idx][round_idx] = valid_smiles[valid_indices.index(idx)]
        else:
            total_oracle_calls += 0  # no valid molecules

        # Update z for next round
        z_current = z_next

    # For each trajectory, find best round
    results = []
    for i in range(num_starts):
        best_round = int(np.argmax(all_scores[i]))
        best_score = all_scores[i, best_round]
        best_smi = all_smiles[i][best_round]
        results.append({
            "trajectory_idx": i,
            "best_round": best_round,
            "best_score": best_score,
            "best_smiles": best_smi,
            "scores_per_round": all_scores[i].tolist(),
        })

    # Sort by score descending
    results.sort(key=lambda x: x["best_score"], reverse=True)

    return {
        "results": results,
        "all_scores": all_scores,
        "total_oracle_calls": total_oracle_calls,
    }

"""Evaluation metrics for autoencoder and inference."""

from __future__ import annotations

from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from routeflow.data.route import Route, TokenType


# ============================================================
#  Helpers
# ============================================================

def _canonical_smiles(smi: str) -> Optional[str]:
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _tanimoto_morgan(smi1: str, smi2: str, radius: int = 2, nbits: int = 4096) -> float:
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _scaffold_similarity(smi1: str, smi2: str) -> float:
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        scf1 = MurckoScaffold.GetScaffoldForMol(mol1)
        scf2 = MurckoScaffold.GetScaffoldForMol(mol2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(scf1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(scf2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


# ============================================================
#  Autoencoder Evaluation Metrics
# ============================================================

def _route_to_step_set(route: Route) -> set[tuple]:
    """Convert a route into a set of (rxn_idx, frozenset(bb_ids)) tuples.

    Extracts from tokens (not steps.sources) for consistency, since
    decode_autoregressive does not populate sources.

    Walks the token sequence with a state machine to group each RXN token
    with the BB tokens that follow it (until the next action or END).
    """
    step_set = set()
    current_rxn = None
    current_bbs = []

    for t in route.tokens:
        if t.token_type in (TokenType.BRANCH, TokenType.REACT):
            # Flush previous step if any
            if current_rxn is not None:
                step_set.add((current_rxn, frozenset(current_bbs)))
                current_rxn = None
                current_bbs = []
        elif t.token_type == TokenType.RXN:
            current_rxn = t.value
            current_bbs = []
        elif t.token_type == TokenType.BB:
            current_bbs.append(t.value)
        elif t.token_type == TokenType.END:
            if current_rxn is not None:
                step_set.add((current_rxn, frozenset(current_bbs)))
                current_rxn = None
                current_bbs = []

    # Flush last step if no END token
    if current_rxn is not None:
        step_set.add((current_rxn, frozenset(current_bbs)))

    return step_set


def _route_token_signature(route: Route) -> tuple:
    """Order-preserving token signature: tuple of (token_type, value) per token.

    Drops START / END so reconstructed routes (which may or may not include END)
    compare cleanly against ground truth. Used for exact-route match.
    """
    sig = []
    for t in route.tokens:
        if t.token_type in (TokenType.START, TokenType.END):
            continue
        v = t.value if t.value is not None else 0
        sig.append((int(t.token_type), int(v)))
    return tuple(sig)


def _route_to_token_multisets(route: Route) -> dict:
    """Extract multisets of actions, rxn IDs, and BB IDs from a route's tokens."""
    actions = []
    rxn_ids = []
    bb_ids = []
    n_pops = 0
    for t in route.tokens:
        if t.token_type in (TokenType.BRANCH, TokenType.REACT):
            actions.append(t.token_type)
        elif t.token_type == TokenType.RXN:
            rxn_ids.append(t.value)
        elif t.token_type == TokenType.BB:
            bb_ids.append(t.value)
        elif t.token_type == TokenType.POP:
            n_pops += 1
    return {
        "actions": sorted(actions),
        "rxn_ids": sorted(rxn_ids),
        "bb_ids": sorted(bb_ids),
        "n_pops": n_pops,
    }


def compute_ae_metrics(
    original_routes: list[Route],
    reconstructed_routes: list[Optional[Route]],
) -> dict:
    """Compute autoencoder evaluation metrics.

    Product-Level:
        validity:          decode succeeded (not None)
        product_match:     final product canonical SMILES exact match
        exact_route:       full token sequence match AND product canonical match
                           (strict subset of product_match; same recipe + same product)
        tanimoto_sim:      avg Morgan Tanimoto similarity (excluding exact matches)
        tanimoto_sim_all:  avg Morgan Tanimoto similarity (all valid pairs)
        scaffold_sim:      avg Murcko scaffold similarity (excluding exact matches)
        scaffold_sim_all:  avg Murcko scaffold similarity (all valid pairs)

    Step-Level (set-based):
        action_seq_acc:    action multiset exact match (order-invariant)
        rxn_set_acc:       reaction ID multiset exact match
        bb_set_acc:        BB ID multiset exact match
        pop_count_acc:     pop count exact match
        step_set_acc:      full step set match (rxn + BBs per step)
        rxn_recall:        fraction of original rxn IDs present in reconstruction
        bb_recall:         fraction of original BB IDs present in reconstruction
    """
    n = len(original_routes)

    # Product-level
    n_valid = 0
    n_product_match = 0
    n_exact_route = 0
    tanimoto_sims_all = []
    scaffold_sims_all = []
    tanimoto_sims_nonmatch = []
    scaffold_sims_nonmatch = []

    # Step-level (set-based)
    n_action_seq_match = 0
    n_rxn_set_match = 0
    n_bb_set_match = 0
    n_pop_count_match = 0
    n_step_set_match = 0
    rxn_recalls = []
    bb_recalls = []
    n_compared = 0  # valid pairs for step-level comparison

    for orig, recon in zip(original_routes, reconstructed_routes):
        if recon is None:
            tanimoto_sims_all.append(0.0)
            scaffold_sims_all.append(0.0)
            continue

        # Validity
        n_valid += 1

        # Product comparison
        orig_canon = _canonical_smiles(orig.final_product) if orig.final_product else None
        recon_canon = _canonical_smiles(recon.final_product) if recon.final_product else None

        is_exact_match = False
        if orig_canon and recon_canon:
            if orig_canon == recon_canon:
                n_product_match += 1
                is_exact_match = True
            tan = _tanimoto_morgan(orig.final_product, recon.final_product)
            scf = _scaffold_similarity(orig.final_product, recon.final_product)
            tanimoto_sims_all.append(tan)
            scaffold_sims_all.append(scf)
            if not is_exact_match:
                tanimoto_sims_nonmatch.append(tan)
                scaffold_sims_nonmatch.append(scf)
        else:
            tanimoto_sims_all.append(0.0)
            scaffold_sims_all.append(0.0)

        # Exact route match: identical token sequence AND identical canonical
        # product. The product check is needed because REACT slot assignment is
        # not part of the token stream — same tokens can yield different
        # products depending on which matched reactant slot the stack-top is
        # routed to. Requiring product_match guarantees exact_route ⊆ product_match.
        if is_exact_match and _route_token_signature(orig) == _route_token_signature(recon):
            n_exact_route += 1

        # Step-level (set-based comparison)
        n_compared += 1
        orig_ms = _route_to_token_multisets(orig)
        recon_ms = _route_to_token_multisets(recon)

        # Action multiset match
        if orig_ms["actions"] == recon_ms["actions"]:
            n_action_seq_match += 1

        # Rxn multiset match
        if orig_ms["rxn_ids"] == recon_ms["rxn_ids"]:
            n_rxn_set_match += 1

        # BB multiset match
        if orig_ms["bb_ids"] == recon_ms["bb_ids"]:
            n_bb_set_match += 1

        # Pop count match
        if orig_ms["n_pops"] == recon_ms["n_pops"]:
            n_pop_count_match += 1

        # Full step set match (rxn + BB grouping)
        orig_steps = _route_to_step_set(orig)
        recon_steps = _route_to_step_set(recon)
        if orig_steps == recon_steps:
            n_step_set_match += 1

        # Rxn recall: what fraction of original rxns appear in reconstruction
        orig_rxn_set = set(orig_ms["rxn_ids"])
        recon_rxn_set = set(orig_ms["rxn_ids"])  # dedup for recall
        recon_rxn_set = set(recon_ms["rxn_ids"])
        if orig_rxn_set:
            rxn_recalls.append(len(orig_rxn_set & recon_rxn_set) / len(orig_rxn_set))

        # BB recall
        orig_bb_set = set(orig_ms["bb_ids"])
        recon_bb_set = set(recon_ms["bb_ids"])
        if orig_bb_set:
            bb_recalls.append(len(orig_bb_set & recon_bb_set) / len(orig_bb_set))

    metrics = {
        # Product-level
        "validity": n_valid / max(n, 1),
        "product_match": n_product_match / max(n, 1),
        "exact_route": n_exact_route / max(n, 1),
        "tanimoto_sim_all": float(np.mean(tanimoto_sims_all)) if tanimoto_sims_all else 0.0,
        "tanimoto_sim": float(np.mean(tanimoto_sims_nonmatch)) if tanimoto_sims_nonmatch else 0.0,
        "scaffold_sim_all": float(np.mean(scaffold_sims_all)) if scaffold_sims_all else 0.0,
        "scaffold_sim": float(np.mean(scaffold_sims_nonmatch)) if scaffold_sims_nonmatch else 0.0,
        # Step-level (set-based)
        "action_seq_acc": n_action_seq_match / max(n_compared, 1),
        "rxn_set_acc": n_rxn_set_match / max(n_compared, 1),
        "bb_set_acc": n_bb_set_match / max(n_compared, 1),
        "pop_count_acc": n_pop_count_match / max(n_compared, 1),
        "step_set_acc": n_step_set_match / max(n_compared, 1),
        "rxn_recall": float(np.mean(rxn_recalls)) if rxn_recalls else 0.0,
        "bb_recall": float(np.mean(bb_recalls)) if bb_recalls else 0.0,
    }

    return metrics


def print_ae_metrics(metrics: dict, split: str = "val"):
    """Pretty-print autoencoder metrics."""
    print(f"\n{'='*50}")
    print(f"  Autoencoder Metrics ({split})")
    print(f"{'='*50}")
    print(f"  Product-Level")
    print(f"  {'Validity':<25s} {metrics['validity']*100:6.2f}%")
    print(f"  {'Product match':<25s} {metrics['product_match']*100:6.2f}%")
    print(f"  {'Exact route':<25s} {metrics['exact_route']*100:6.2f}%")
    print(f"  {'Tanimoto (all)':<25s} {metrics['tanimoto_sim_all']:6.4f}")
    print(f"  {'Tanimoto (non-match)':<25s} {metrics['tanimoto_sim']:6.4f}")
    print(f"  {'Scaffold (all)':<25s} {metrics['scaffold_sim_all']:6.4f}")
    print(f"  {'Scaffold (non-match)':<25s} {metrics['scaffold_sim']:6.4f}")
    print(f"  Step-Level (set-based)")
    print(f"  {'Action seq acc':<25s} {metrics['action_seq_acc']*100:6.2f}%")
    print(f"  {'Rxn set acc':<25s} {metrics['rxn_set_acc']*100:6.2f}%")
    print(f"  {'BB set acc':<25s} {metrics['bb_set_acc']*100:6.2f}%")
    print(f"  {'Pop count acc':<25s} {metrics['pop_count_acc']*100:6.2f}%")
    print(f"  {'Step set acc':<25s} {metrics['step_set_acc']*100:6.2f}%")
    print(f"  {'Rxn recall':<25s} {metrics['rxn_recall']*100:6.2f}%")
    print(f"  {'BB recall':<25s} {metrics['bb_recall']*100:6.2f}%")
    print(f"{'='*50}\n")


# ============================================================
#  Inference Metrics
# ============================================================

def compute_inference_metrics(results: list[dict]) -> dict:
    """Compute final inference metrics from optimization results."""
    scores = [r["best_score"] for r in results]
    scores_sorted = sorted(scores, reverse=True)

    n = len(scores_sorted)
    top1 = scores_sorted[0] if n >= 1 else 0.0
    top10 = np.mean(scores_sorted[:10]) if n >= 10 else np.mean(scores_sorted)
    top100 = np.mean(scores_sorted[:100]) if n >= 100 else np.mean(scores_sorted)

    valid_smiles = [r["best_smiles"] for r in results if r["best_smiles"] is not None]
    unique_smiles = set(valid_smiles)

    return {
        "top1": top1,
        "top10_avg": top10,
        "top100_avg": top100,
        "num_valid": len(valid_smiles),
        "num_unique": len(unique_smiles),
        "diversity": len(unique_smiles) / max(len(valid_smiles), 1),
    }


def print_inference_metrics(metrics: dict):
    """Pretty-print inference metrics."""
    print(f"\n{'='*50}")
    print(f"  Inference Results")
    print(f"{'='*50}")
    print(f"  {'Top-1':<20s} {metrics['top1']:.4f}")
    print(f"  {'Top-10 avg':<20s} {metrics['top10_avg']:.4f}")
    print(f"  {'Top-100 avg':<20s} {metrics['top100_avg']:.4f}")
    print(f"  {'Valid molecules':<20s} {metrics['num_valid']}")
    print(f"  {'Unique molecules':<20s} {metrics['num_unique']}")
    print(f"  {'Diversity':<20s} {metrics['diversity']:.4f}")
    print(f"{'='*50}\n")

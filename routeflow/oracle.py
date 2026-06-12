"""Unified oracle interface for all supported properties.

Usage:
    oracle_fn = get_oracle("GSK3B")          # TDC oracle
    oracle_fn = get_oracle("DRD3")           # docking oracle (1K budget)
    scores = oracle_fn(["SMILES1", "SMILES2"])       # returns list of floats
"""

from __future__ import annotations

from typing import Callable

# Map our property name → underlying TDC oracle name (when they differ).
# For docking oracles, our short name is the receptor (DRD3 etc.); TDC uses
# PDB-based names. Property names not in this dict are passed to TDC verbatim.
TDC_ORACLE_NAME_MAP = {
    "DRD3": "3pbl_docking_normalize",
    "EGFR": "2rgp_docking_normalize",
    "A2AR": "3eml_docking_normalize",
}

# Recommended oracle-call budget per property (used by training scripts).
# Docking oracles are expensive (~2s/mol), so default to 1000 instead of 10K.
ORACLE_BUDGET = {
    "DRD3": 1000,
    "EGFR": 1000,
    "A2AR": 1000,
}
DEFAULT_ORACLE_BUDGET = 10000


def get_oracle_budget(name: str) -> int:
    """Return recommended oracle-call budget for a property."""
    return ORACLE_BUDGET.get(name, DEFAULT_ORACLE_BUDGET)


# All supported properties
SUPPORTED_PROPERTIES = [
    # 7 MPO tasks
    "amlodipine_mpo",
    "fexofenadine_mpo",
    "osimertinib_mpo",
    "perindopril_mpo",
    "ranolazine_mpo",
    "sitagliptin_mpo",
    "zaleplon_mpo",
    # 3 similarity tasks
    "celecoxib_rediscovery",
    "median_1",
    "median_2",
    # 3 protein-binding tasks (TDC)
    "JNK3",
    "DRD2",
    "GSK3B",
    # 3 docking-based protein-binding tasks (TDC; 1K oracle budget)
    "DRD3",
    "EGFR",
    "A2AR",
]


def get_oracle(name: str) -> Callable:
    """Return a callable oracle: list[str] → list[float].

    For TDC oracles, wraps tdc.Oracle.
    Docking-based receptor oracles (DRD3, EGFR, A2AR) are mapped to TDC
    PDB-based names via TDC_ORACLE_NAME_MAP.
    """
    if name not in SUPPORTED_PROPERTIES:
        raise ValueError(
            f"Unknown property '{name}'. Supported: {SUPPORTED_PROPERTIES}"
        )

    return _build_tdc_oracle(name)


def _build_tdc_oracle(name: str) -> Callable:
    """Build a TDC oracle that accepts list[str] → list[float].

    Sanitizes NaN/Inf scores to 0.0 so docking failures don't corrupt training.
    """
    import math
    from tdc import Oracle
    tdc_name = TDC_ORACLE_NAME_MAP.get(name, name)
    tdc_oracle = Oracle(name=tdc_name)

    def oracle_fn(smiles_list: list[str]) -> list[float]:
        scores = tdc_oracle(smiles_list)
        if not isinstance(scores, list):
            scores = [scores]
        out = []
        for s in scores:
            try:
                v = float(s)
                if math.isnan(v) or math.isinf(v):
                    v = 0.0
            except (TypeError, ValueError):
                v = 0.0
            out.append(v)
        return out

    return oracle_fn

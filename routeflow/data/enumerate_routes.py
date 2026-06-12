"""Random enumeration of valid synthesis routes (Branch/React/Pop, depth ≤ 3, ≤ 6 rxns)."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm

# Suppress RDKit warnings (kekulization, valence errors) — these are handled by try/except
RDLogger.logger().setLevel(RDLogger.ERROR)

from routeflow.chem.compatibility import CompatibilityMatrix
from routeflow.chem.executor import ReactionExecutor, SynthesisStack
from routeflow.chem.template import Reaction
from routeflow.data.route import Route, RouteStep, RouteToken, TokenType


def _pick_random(arr: np.ndarray, rng: random.Random) -> int:
    """Pick a random element from a numpy array."""
    return int(arr[rng.randint(0, len(arr) - 1)])


def _try_branch(
    compat: CompatibilityMatrix,
    rng: random.Random,
) -> Optional[tuple[int, list[int], Chem.Mol, str]]:
    """Try to create a Branch step: pick seed reaction + all BBs.

    Returns (rxn_idx, bb_indices, product_mol, product_smiles) or None.
    """
    seed_rxns = compat.get_seed_reactions()
    if len(seed_rxns) == 0:
        return None

    rxn_idx = _pick_random(seed_rxns, rng)
    rxn = compat.reactions[rxn_idx]
    n_reactants = rxn.num_reactants

    # Pick one compatible BB per position
    bb_indices = []
    reactant_mols = []
    for pos in range(n_reactants):
        candidates = compat.get_compatible_bbs(rxn_idx, pos)
        if len(candidates) == 0:
            return None
        bb_idx = _pick_random(candidates, rng)
        bb_indices.append(bb_idx)
        reactant_mols.append(compat.bb_mols[bb_idx])

    product = ReactionExecutor.execute(rxn, reactant_mols)
    if product is None:
        return None

    smi = Chem.MolToSmiles(product, canonical=True)
    return rxn_idx, bb_indices, product, smi


def _try_react(
    compat: CompatibilityMatrix,
    stack: SynthesisStack,
    rng: random.Random,
    allow_pop: bool = True,
    prefer_pop: bool = False,
    require_multi_reactant: bool = False,
) -> Optional[tuple[int, list[dict], Chem.Mol, str]]:
    """Try to create a React step: pop stack top, pick reaction, fill remaining positions.

    Args:
        allow_pop: whether Pop is allowed as a source
        prefer_pop: if True, strongly prefer Pop over BB (for merge steps)

    Returns (rxn_idx, sources, product_mol, product_smiles) or None.
    sources: list of {"type": "bb"|"pop", "bb_idx": int|None}
    """
    if stack.is_empty:
        return None

    mol_top = stack.top()
    if mol_top is None:
        return None

    # Find reactions that match mol_top at some position
    rxn_pos_map = compat.get_bu_rxn_mask_with_positions(mol_top)
    if not rxn_pos_map:
        return None

    # For merge OR last-step extend: filter to multi-reactant reactions only
    # (uni-mol React emits no source tokens, so the last token would be RXN
    # which violates the END-must-follow-BB/POP rule).
    if prefer_pop or require_multi_reactant:
        multi_rxns = {k: v for k, v in rxn_pos_map.items()
                      if compat.reactions[k].num_reactants >= 2}
        if multi_rxns:
            rxn_pos_map = multi_rxns
        else:
            return None  # no multi-reactant reaction available

    # Randomly pick a reaction
    rxn_idx = rng.choice(list(rxn_pos_map.keys()))
    rxn = compat.reactions[rxn_idx]
    matched_positions = rxn_pos_map[rxn_idx]
    n_reactants = rxn.num_reactants

    if n_reactants == 1:
        # Uni-molecular: mol_top is the sole reactant
        pos0 = matched_positions[0]
        product = ReactionExecutor.execute(rxn, [mol_top])
        if product is None:
            return None
        smi = Chem.MolToSmiles(product, canonical=True)
        sources = []
        return rxn_idx, sources, product, smi

    # mol_top is always routed to slot 0 (BU mask already filtered to
    # reactions where 0 is in matched_positions).
    pos0 = 0

    # Fill remaining positions
    remaining_positions = [p for p in range(n_reactants) if p != pos0]
    sources = []
    reactant_mols = [None] * n_reactants
    reactant_mols[pos0] = mol_top

    # Temporary copy of the stack to handle Pop
    temp_stack = stack.copy()
    temp_stack.pop()  # remove mol_top

    pop_prob = 0.8 if prefer_pop else 0.3

    for pos in remaining_positions:
        can_pop = (
            allow_pop
            and not temp_stack.is_empty
            and rxn.reactant_templates[pos].match(temp_stack.top())
        )
        candidates_bb = compat.get_compatible_bbs(rxn_idx, pos)
        has_bb = len(candidates_bb) > 0

        if can_pop and has_bb:
            use_pop = rng.random() < pop_prob
        elif can_pop:
            use_pop = True
        elif has_bb:
            use_pop = False
        else:
            return None

        if use_pop:
            pop_mol, _ = temp_stack.pop()
            reactant_mols[pos] = pop_mol
            sources.append({"type": "pop", "bb_idx": None})
        else:
            bb_idx = _pick_random(candidates_bb, rng)
            reactant_mols[pos] = compat.bb_mols[bb_idx]
            sources.append({"type": "bb", "bb_idx": bb_idx})

    # Execute reaction
    product = ReactionExecutor.execute(rxn, reactant_mols)
    if product is None:
        return None

    smi = Chem.MolToSmiles(product, canonical=True)
    return rxn_idx, sources, product, smi


def _build_tokens_for_branch(rxn_idx: int, bb_indices: list[int]) -> list[RouteToken]:
    """Build token list for a Branch step: [BR, rxn, bb0, bb1, ...]."""
    tokens = [
        RouteToken(TokenType.BRANCH),
        RouteToken(TokenType.RXN, rxn_idx),
    ]
    for bb_idx in bb_indices:
        tokens.append(RouteToken(TokenType.BB, bb_idx))
    return tokens


def _build_tokens_for_react(rxn_idx: int, sources: list[dict]) -> list[RouteToken]:
    """Build token list for a React step: [RE, rxn, source0, source1, ...].

    Note: pos0 (stack top) is implicit (always popped), not in token list.
    """
    tokens = [
        RouteToken(TokenType.REACT),
        RouteToken(TokenType.RXN, rxn_idx),
    ]
    for src in sources:
        if src["type"] == "pop":
            tokens.append(RouteToken(TokenType.POP))
        else:
            tokens.append(RouteToken(TokenType.BB, src["bb_idx"]))
    return tokens


def _add_branch_step(
    compat: CompatibilityMatrix,
    stack: SynthesisStack,
    route: Route,
    rng: random.Random,
    max_heavy_atoms: int,
    max_attempts: int = 10,
) -> bool:
    """Try to add a Branch step. Returns True on success."""
    for _ in range(max_attempts):
        result = _try_branch(compat, rng)
        if result is not None:
            rxn_idx, bb_indices, product, product_smi = result
            if ReactionExecutor.num_heavy_atoms(product) <= max_heavy_atoms:
                tokens = _build_tokens_for_branch(rxn_idx, bb_indices)
                route.tokens.extend(tokens)
                stack.push(product, product_smi)
                sources = [{"type": "bb", "bb_idx": idx} for idx in bb_indices]
                route.steps.append(RouteStep(
                    action="branch", rxn_idx=rxn_idx,
                    sources=sources, product_smiles=product_smi,
                ))
                route.intermediates.append(product_smi)
                return True
    return False


def _add_react_merge_step(
    compat: CompatibilityMatrix,
    stack: SynthesisStack,
    route: Route,
    rng: random.Random,
    max_heavy_atoms: int,
    max_attempts: int = 20,
) -> bool:
    """Try to add a React step that merges stack top with Pop (consuming 2 stack items).

    Returns True on success.
    """
    if stack.depth < 2:
        return False

    for _ in range(max_attempts):
        result = _try_react(compat, stack, rng, allow_pop=True, prefer_pop=True)
        if result is not None:
            rxn_idx, sources, product, product_smi = result
            if ReactionExecutor.num_heavy_atoms(product) <= max_heavy_atoms:
                has_pop = any(s["type"] == "pop" for s in sources)
                if not has_pop:
                    continue  # we need a Pop to actually merge, retry
                tokens = _build_tokens_for_react(rxn_idx, sources)
                route.tokens.extend(tokens)
                stack.pop()  # pop mol_top (pos0)
                for src in sources:
                    if src["type"] == "pop":
                        stack.pop()
                stack.push(product, product_smi)
                route.steps.append(RouteStep(
                    action="react", rxn_idx=rxn_idx,
                    sources=sources, product_smiles=product_smi,
                ))
                route.intermediates.append(product_smi)
                return True
    return False


def _add_react_extend_step(
    compat: CompatibilityMatrix,
    stack: SynthesisStack,
    route: Route,
    rng: random.Random,
    max_heavy_atoms: int,
    max_attempts: int = 10,
) -> bool:
    """Try to add a React step that extends stack top (no Pop, uses BB or uni-mol).

    Does not change stack depth (pops 1, pushes 1).
    """
    if stack.is_empty:
        return False

    for _ in range(max_attempts):
        result = _try_react(compat, stack, rng, allow_pop=False)
        if result is not None:
            rxn_idx, sources, product, product_smi = result
            if ReactionExecutor.num_heavy_atoms(product) <= max_heavy_atoms:
                tokens = _build_tokens_for_react(rxn_idx, sources)
                route.tokens.extend(tokens)
                stack.pop()
                stack.push(product, product_smi)
                route.steps.append(RouteStep(
                    action="react", rxn_idx=rxn_idx,
                    sources=sources, product_smiles=product_smi,
                ))
                route.intermediates.append(product_smi)
                return True
    return False


def enumerate_single_route(
    compat: CompatibilityMatrix,
    max_tree_depth: int = 3,
    min_tree_depth: int = 1,
    max_heavy_atoms: int = 80,
    max_reactions: int = 5,
    rng: Optional[random.Random] = None,
    max_attempts_per_step: int = 10,
) -> Optional[Route]:
    """Enumerate a single valid synthesis route with guaranteed convergence.

    Structured strategy that ensures stack.depth == 1 at the end:
      depth=1: 1 Branch (optionally + extend React)
      depth=2: 2 Branches → 1 merge React (optionally + extend React)
      depth=3: 2+ Branches → merge → 2+ items → merge again

    Returns Route or None if failed.
    """
    if rng is None:
        rng = random.Random()

    stack = SynthesisStack()
    route = Route()
    route.tokens = [RouteToken(TokenType.START)]

    target_depth = rng.randint(min_tree_depth, max_tree_depth)

    if target_depth == 1:
        # ---- Depth 1: single branch + optional extend ----
        if not _add_branch_step(compat, stack, route, rng, max_heavy_atoms):
            return None
        # Optionally extend with React (uni-mol or bi-mol with BB, no Pop)
        n_extends = rng.randint(0, 2)
        for _ in range(n_extends):
            _add_react_extend_step(compat, stack, route, rng, max_heavy_atoms)

    elif target_depth == 2:
        # ---- Depth 2: N branches → merge until stack=1 ----
        num_branches = rng.randint(2, 4)
        created = 0
        for _ in range(num_branches):
            if _add_branch_step(compat, stack, route, rng, max_heavy_atoms):
                created += 1
                # Optionally extend each branch
                if rng.random() < 0.3:
                    _add_react_extend_step(compat, stack, route, rng, max_heavy_atoms)
        if created < 2:
            return None

        # Merge until stack has 1 item
        merge_failures = 0
        while stack.depth > 1 and merge_failures < 5:
            if _add_react_merge_step(compat, stack, route, rng, max_heavy_atoms):
                merge_failures = 0
            else:
                merge_failures += 1
        if stack.depth != 1:
            return None

    elif target_depth >= 3:
        # ---- Depth 3/4/5: build N sub-trees, merge iteratively ----
        # Number of sub-trees scales with depth
        num_subtrees = target_depth - 1  # depth 3→2, depth 4→3, depth 5→4

        for st in range(num_subtrees):
            # Each sub-tree: branch + optional extend
            n_branches = rng.randint(1, 3) if st > 0 else rng.randint(2, 3)
            created = 0
            for _ in range(n_branches):
                if _add_branch_step(compat, stack, route, rng, max_heavy_atoms):
                    created += 1
                    if rng.random() < 0.3:
                        _add_react_extend_step(compat, stack, route, rng, max_heavy_atoms)
            if created < 1:
                return None

            # Merge this sub-tree if stack depth > 1 (except after last sub-tree)
            if st < num_subtrees - 1:
                merge_failures = 0
                while stack.depth > 1 and merge_failures < 5:
                    if _add_react_merge_step(compat, stack, route, rng, max_heavy_atoms):
                        merge_failures = 0
                    else:
                        merge_failures += 1
                if stack.depth != 1:
                    return None

        # Final merge: merge all remaining items
        merge_failures = 0
        while stack.depth > 1 and merge_failures < 5:
            if _add_react_merge_step(compat, stack, route, rng, max_heavy_atoms):
                merge_failures = 0
            else:
                merge_failures += 1
        if stack.depth != 1:
            return None

    # Final check
    if stack.depth != 1:
        return None

    route.tokens.append(RouteToken(TokenType.END))
    route.num_reactions = len(route.steps)
    route.tree_depth = target_depth
    route.final_product = stack.top_smiles()

    # Reject if too many reactions
    if route.num_reactions > max_reactions:
        return None

    return route


# ============================================================
# Rxn-count-targeted enumeration (used by current pipeline)
# ============================================================


def _action_feasible(stack_d: int, rxns_left: int) -> bool:
    """Can we reach stack.depth=1 in exactly rxns_left more reactions?"""
    if rxns_left < 0 or stack_d < 0:
        return False
    if rxns_left == 0:
        return stack_d == 1
    b_min = max(0, 1 - stack_d)
    b_max = (rxns_left + 1 - stack_d) // 2
    return b_min <= b_max


def _add_react_extend_step_safe(
    compat, stack, route, rng, max_heavy_atoms,
    require_multi_reactant: bool = False,
    max_attempts: int = 10,
) -> bool:
    """Variant of _add_react_extend_step that can require multi-reactant rxns.

    Used for the last reaction in a route so the trailing token is BB (not RXN).
    """
    if stack.is_empty:
        return False
    for _ in range(max_attempts):
        result = _try_react(
            compat, stack, rng, allow_pop=False,
            require_multi_reactant=require_multi_reactant,
        )
        if result is not None:
            rxn_idx, sources, product, product_smi = result
            if ReactionExecutor.num_heavy_atoms(product) <= max_heavy_atoms:
                tokens = _build_tokens_for_react(rxn_idx, sources)
                route.tokens.extend(tokens)
                stack.pop()
                stack.push(product, product_smi)
                route.steps.append(RouteStep(
                    action="react", rxn_idx=rxn_idx,
                    sources=sources, product_smiles=product_smi,
                ))
                route.intermediates.append(product_smi)
                return True
    return False


def enumerate_single_route_by_rxn_count(
    compat: CompatibilityMatrix,
    target_rxns: int,
    max_heavy_atoms: int = 80,
    rng: Optional[random.Random] = None,
) -> Optional[Route]:
    """Build a single route with exactly `target_rxns` reactions.

    Action selection at each step is restricted to (branch / react_extend /
    react_merge) options whose post-state still admits a finishing path to
    stack.depth=1. The last reaction is forced multi-reactant so the route
    ends on a BB or POP token (END-validation requirement).
    """
    if rng is None:
        rng = random.Random()

    stack = SynthesisStack()
    route = Route()
    route.tokens = [RouteToken(TokenType.START)]
    rxns_left = target_rxns

    while rxns_left > 0:
        is_last = (rxns_left == 1)
        candidates = []
        if _action_feasible(stack.depth + 1, rxns_left - 1):
            candidates.append("branch")
        if stack.depth >= 1 and _action_feasible(stack.depth, rxns_left - 1):
            candidates.append("react_extend")
        if stack.depth >= 2 and _action_feasible(stack.depth - 1, rxns_left - 1):
            candidates.append("react_merge")

        if not candidates:
            return None

        rng.shuffle(candidates)
        progressed = False
        for action in candidates:
            if action == "branch":
                ok = _add_branch_step(compat, stack, route, rng, max_heavy_atoms)
            elif action == "react_extend":
                ok = _add_react_extend_step_safe(
                    compat, stack, route, rng, max_heavy_atoms,
                    require_multi_reactant=is_last,
                )
            else:  # react_merge
                ok = _add_react_merge_step(compat, stack, route, rng, max_heavy_atoms)
            if ok:
                progressed = True
                break

        if not progressed:
            return None
        rxns_left -= 1

    # Validate end conditions
    if stack.depth != 1:
        return None
    if not route.tokens or route.tokens[-1].token_type not in (TokenType.BB, TokenType.POP):
        return None

    route.tokens.append(RouteToken(TokenType.END))
    route.num_reactions = len(route.steps)
    route.tree_depth = 0
    route.final_product = stack.top_smiles()
    return route


def _worker_enumerate_by_rxn_count(args):
    """Worker that generates `num_routes` routes with exactly `target_rxns` reactions."""
    compat, target_rxns, num_routes, max_heavy_atoms, seed, max_attempts = args
    rng = random.Random(seed)
    routes = []
    attempts = 0
    while len(routes) < num_routes and attempts < max_attempts:
        attempts += 1
        r = enumerate_single_route_by_rxn_count(
            compat=compat,
            target_rxns=target_rxns,
            max_heavy_atoms=max_heavy_atoms,
            rng=rng,
        )
        if r is not None and r.final_product is not None:
            routes.append(r)
    return routes


def enumerate_routes_by_rxn_count(
    compat: CompatibilityMatrix,
    num_routes: int,
    target_rxns: int,
    max_heavy_atoms: int = 80,
    seed: int = 42,
    num_workers: int = 32,
    max_attempts_factor: int = 50,
) -> list[Route]:
    """Multiprocessing wrapper: generate `num_routes` routes with exactly
    `target_rxns` reactions. No dedup (caller's responsibility)."""
    from multiprocessing import Pool

    if num_workers <= 1:
        return _worker_enumerate_by_rxn_count(
            (compat, target_rxns, num_routes, max_heavy_atoms, seed,
             num_routes * max_attempts_factor)
        )

    per_worker = (num_routes + num_workers - 1) // num_workers
    args_list = [
        (compat, target_rxns, per_worker, max_heavy_atoms,
         seed + i * 1000, per_worker * max_attempts_factor)
        for i in range(num_workers)
    ]
    with Pool(num_workers) as p:
        results = p.map(_worker_enumerate_by_rxn_count, args_list)
    return [r for batch in results for r in batch][:num_routes]


def _build_quality_filter(sa_max=None, qed_min=None, sa_min=None):
    """Build a per-route filter closure.

    Conditions (all must hold if specified):
      - SA < sa_max   (upper bound on synthetic accessibility — easy set)
      - SA > sa_min   (lower bound — hard set, OOD direction)
      - QED > qed_min (drug-likeness lower bound)

    Returns None if all thresholds are None. RDKit + sascorer are imported
    lazily so each worker only pays the import cost once.
    """
    if sa_max is None and qed_min is None and sa_min is None:
        return None
    from rdkit import Chem
    from rdkit.Chem import QED, RDConfig
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer

    def _f(route):
        smi = getattr(route, "final_product", None)
        if not smi:
            return False
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        try:
            if qed_min is not None and QED.qed(mol) <= qed_min:
                return False
            if sa_max is not None or sa_min is not None:
                sa = sascorer.calculateScore(mol)
                if sa_max is not None and sa >= sa_max:
                    return False
                if sa_min is not None and sa <= sa_min:
                    return False
        except Exception:
            return False
        return True
    return _f


def _worker_enumerate(args):
    """Worker function for multiprocessing route enumeration.

    If `sa_max` and/or `qed_min` are provided, the worker applies the SA/QED
    filter inline (parallel) and only counts/returns passing routes — this
    eliminates the single-threaded filter bottleneck in the parent process.
    """
    (compat, num_routes, max_tree_depth, min_tree_depth, max_heavy_atoms,
     max_reactions, seed, max_global_attempts, counter, lock,
     sa_max, qed_min, sa_min) = args
    rng = random.Random(seed)
    routes = []
    attempts = 0

    filter_fn = _build_quality_filter(sa_max=sa_max, qed_min=qed_min, sa_min=sa_min)

    while len(routes) < num_routes and attempts < num_routes * max_global_attempts:
        attempts += 1
        route = enumerate_single_route(
            compat=compat,
            max_tree_depth=max_tree_depth,
            min_tree_depth=min_tree_depth,
            max_heavy_atoms=max_heavy_atoms,
            max_reactions=max_reactions,
            rng=rng,
        )
        if route is None or route.final_product is None:
            continue
        if filter_fn is not None and not filter_fn(route):
            continue
        routes.append(route)
        if counter is not None:
            with lock:
                counter.value += 1

    return routes


def enumerate_routes(
    compat: CompatibilityMatrix,
    num_routes: int,
    max_tree_depth: int = 3,
    min_tree_depth: int = 1,
    max_heavy_atoms: int = 80,
    max_reactions: int = 8,
    seed: int = 42,
    max_global_attempts: int = 100,
    num_workers: int = 32,
    deduplicate: bool = False,
    seen_products: set = None,
    sa_max: float = None,
    qed_min: float = None,
    sa_min: float = None,
) -> list[Route]:
    """Enumerate multiple valid synthesis routes using multiprocessing.

    Args:
        deduplicate: if True, ensure all routes have unique final product SMILES.
        seen_products: set of already-seen product SMILES (for cross-split dedup).
                       Will NOT be modified; caller should update after.
    """
    target = num_routes
    all_routes = []
    _seen = set(seen_products) if seen_products else set()
    round_num = 0

    while len(all_routes) < target:
        round_num += 1
        remaining = target - len(all_routes)
        # First round: generate exact amount; later rounds: overshoot to compensate dedup
        if deduplicate and round_num > 1:
            to_generate = int(remaining * 1.2)
        else:
            to_generate = remaining

        if round_num > 1:
            print(f"  Round {round_num}: need {remaining} more unique routes, generating {to_generate}...")

        raw_routes = _enumerate_batch(
            compat, to_generate, max_tree_depth, min_tree_depth, max_heavy_atoms, max_reactions,
            seed + round_num * 100000, max_global_attempts, num_workers,
            sa_max=sa_max, qed_min=qed_min, sa_min=sa_min,
        )

        if deduplicate:
            for r in raw_routes:
                if r.final_product and r.final_product not in _seen:
                    _seen.add(r.final_product)
                    all_routes.append(r)
                    if len(all_routes) >= target:
                        break
            print(f"  Unique so far: {len(all_routes)}/{target}")
        else:
            all_routes.extend(raw_routes)

        if not deduplicate:
            break

    all_routes = all_routes[:target]
    print(f"Generated {len(all_routes)} routes" +
          (f" ({len(seen_products)} unique products)" if deduplicate else ""))
    return all_routes


def _enumerate_batch(
    compat: CompatibilityMatrix,
    num_routes: int,
    max_tree_depth: int,
    min_tree_depth: int,
    max_heavy_atoms: int,
    max_reactions: int,
    seed: int,
    max_global_attempts: int,
    num_workers: int,
    sa_max: float = None,
    qed_min: float = None,
    sa_min: float = None,
) -> list[Route]:
    """Generate a batch of routes using multiprocessing.

    If sa_max/qed_min/sa_min are set, the filter is applied inside each worker
    (parallel). Returned routes are already filtered.
    """
    filter_fn_main = (
        _build_quality_filter(sa_max=sa_max, qed_min=qed_min, sa_min=sa_min)
        if num_workers <= 1 else None
    )

    if num_workers <= 1:
        rng = random.Random(seed)
        routes = []
        attempts = 0
        pbar = tqdm(total=num_routes, desc="Enumerating routes")
        while len(routes) < num_routes and attempts < num_routes * max_global_attempts:
            attempts += 1
            route = enumerate_single_route(
                compat=compat,
                max_tree_depth=max_tree_depth,
                min_tree_depth=min_tree_depth,
                max_heavy_atoms=max_heavy_atoms,
                max_reactions=max_reactions,
                rng=rng,
            )
            if route is None or route.final_product is None:
                continue
            if filter_fn_main is not None and not filter_fn_main(route):
                continue
            routes.append(route)
            pbar.update(1)
        pbar.close()
        return routes

    from multiprocessing import Pool, Manager
    import time

    manager = Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    routes_per_worker = num_routes // num_workers
    remainder = num_routes % num_workers

    worker_args = []
    for i in range(num_workers):
        n = routes_per_worker + (1 if i < remainder else 0)
        worker_args.append((
            compat, n, max_tree_depth, min_tree_depth, max_heavy_atoms, max_reactions,
            seed + i * 10000, max_global_attempts, counter, lock,
            sa_max, qed_min, sa_min,
        ))

    print(f"Enumerating {num_routes} routes with {num_workers} workers...", flush=True)

    pool = Pool(processes=num_workers)
    async_results = [pool.apply_async(_worker_enumerate, (a,)) for a in worker_args]

    pbar = tqdm(total=num_routes, desc="Enumerating routes")
    while not all(r.ready() for r in async_results):
        time.sleep(2)
        current = counter.value
        pbar.n = min(current, num_routes)
        pbar.refresh()

    pbar.n = min(counter.value, num_routes)
    pbar.refresh()
    pbar.close()

    results = [r.get() for r in async_results]
    pool.close()
    pool.join()

    routes = []
    for r in results:
        routes.extend(r)

    return routes[:num_routes]

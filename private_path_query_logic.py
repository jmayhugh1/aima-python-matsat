from utils4e import Expr
from logic4e import implies, to_cnf, conjuncts, disjuncts
from typing import List, Dict, Tuple
import math
import numpy as np
from private_path_query_utils import (
    EdgeState,
    Edge,
    PrivatePathInfo,
    normalize_edge_weights,
)


DEFAULT_HARD_CLAUSE_WEIGHT = 1.0
SOFT_CLAUSE_WEIGHT = 0.3


def compute_hard_clause_weight(num_parties: int) -> float:
    """
    Compute hard clause weight based on number of parties.

    Hard weight = num_parties + 1, ensuring hard clauses always dominate
    the total soft clause budget (which is ~1.0 per Bob party).

    Example: Alice + 2 Bobs = 3 parties → hard weight = 4.0
    """
    return float(num_parties + 1)


# HARD EXPRESSIONS: Must be satisfied


def At(t: int, v: int) -> Expr:
    return Expr("At", t, v)


def Move(t: int, u: int, v: int) -> Expr:
    return Expr("Move", t, u, v)


def Wait(t: int, v: int) -> Expr:
    return Expr("Wait", t, v)


def PosBit(t: int, k: int) -> Expr:
    """
    Bit-level position atom.

    Semantics:
      - `PosBit(t, k)` is True iff the k-th bit of the agent's vertex id at time `t`
        is 1, where `k=0` is the least-significant bit.
      - The full position at time `t` is reconstructed by reading all bits:
            pos(t) = sum(2**k for k if PosBit(t, k) is True)

    Example for V=5 -> b=3 bits:
      - vertex 3 (binary 011) means:
            PosBit(t,0)=True, PosBit(t,1)=True, PosBit(t,2)=False
    """
    return Expr("PosBit", t, k)


def Active(u: int, v: int) -> Expr:
    return Expr("Active", u, v)


# SOFT EXPRESSIONS: allowed to be flipped if it means satisfying a hard expression


def Allowed(u: int, v: int) -> Expr:
    return Expr("Allowed", u, v)


def directed_pairs_from_edges(edges: List[Edge], V: int) -> List[Tuple[int, int]]:
    """
    Deterministic directed pair domain induced by edges.
    """
    pairs = sorted(
        {
            (edge.vertex1.id, edge.vertex2.id)
            for edge in edges
            if edge.vertex1.id != edge.vertex2.id
        }
    )
    for u, v in pairs:
        if not (0 <= u < V and 0 <= v < V):
            raise ValueError("Edge vertex id out of range for V")
    return pairs


def num_bits(V: int) -> int:
    """
    Number of bits required to encode vertex ids in [0, V-1].

    We use the minimum fixed width `b = ceil(log2(V))`, with a floor of 1.
    Codes in [V, 2**b - 1] are invalid and are excluded by range clauses.
    """
    if V <= 0:
        raise ValueError("V must be positive")
    return max(1, math.ceil(math.log2(V)))


def _exactly_one(vars: List[Expr]) -> List[Expr]:
    """Return CNF enforcing exactly one of the given literals is True."""
    clauses: List[Expr] = []
    if not vars:
        return clauses

    # At least one: v1 ∨ v2 ∨ ... ∨ vn
    disj = vars[0]
    for lit in vars[1:]:
        disj = disj | lit
    clauses.append(disj)

    # At most one: ¬vi ∨ ¬vj for all i < j
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            clauses.append((~vars[i]) | (~vars[j]))
    return clauses


def forbid_code(t: int, c: int, b: int) -> Expr:
    """
    Return one clause that forbids `pos(t) == c` for a b-bit encoding.

    If `c` has target bits l_0..l_(b-1), this builds:
      (~l_0 OR ~l_1 OR ... OR ~l_(b-1))
    so the exact codeword `c` cannot be assigned.
    """
    clause = None
    for k in range(b):
        bit = (c >> k) & 1
        lit = PosBit(t, k) if bit == 1 else ~PosBit(t, k)
        term = ~lit
        clause = term if clause is None else (clause | term)
    assert clause is not None
    return clause


def force_pos_const(t: int, v: int, b: int) -> List[Expr]:
    """
    Unit clauses fixing `pos(t)` to constant vertex id `v`.

    This is the bit-encoded replacement for one-hot constraints like `At(t, v)`.
    """
    clauses: List[Expr] = []
    for k in range(b):
        bit = (v >> k) & 1
        clauses.append(PosBit(t, k) if bit == 1 else ~PosBit(t, k))
    return clauses


def eq_pos_const(t: int, u: int, b: int) -> Expr:
    """
    Expression encoding the equality test `pos(t) == u`.

    This is a conjunction over bits, not a new SAT variable:
      AND_k (PosBit(t,k) if bit_k(u)=1 else ~PosBit(t,k))
    """
    conj = None
    for k in range(b):
        bit = (u >> k) & 1
        lit = PosBit(t, k) if bit == 1 else ~PosBit(t, k)
        conj = lit if conj is None else (conj & lit)
    assert conj is not None
    return conj


def physics_bits(
    T: int, V: int, directed_pairs: List[Tuple[int, int]] | None = None
) -> List[Expr]:
    """
    Position-bit dynamics for a single agent moving on a directed graph.

    Args:
        T: time horizon (number of steps).
        V: number of vertices (assumed to be 0..V-1).

    Returns:
        List of formulas encoding:
          - bit semantics: PosBit(t, k) collectively represent vertex id at time t
          - range restriction for position bits (disallow invalid vertex codes)
          - transition feasibility: pos(t)=u -> OR_{v in adj[u]} pos(t+1)=v
          - edge legality: (pos(t)=u & pos(t+1)=v) -> Allowed(u,v) for directed edges
          - edge consistency: Allowed(u,v) -> Active(u,v)

    Notes:
      - Wait is implicit via self-loop successor `u -> u`.
      - No one-hot `At/Move/Wait` variables are required in the core encoding.
    """
    formulas: List[Expr] = []
    directed_pairs = directed_pairs or [
        (u, v) for u in range(V) for v in range(V) if u != v
    ]
    b = num_bits(V)

    # 1) Range restriction: disallow codes >= V.
    max_code = 1 << b
    for t in range(T + 1):
        for c in range(V, max_code):
            formulas.append(forbid_code(t, c, b))

    # Build adjacency domain and include wait transitions.
    adj: Dict[int, List[int]] = {u: [] for u in range(V)}
    for u, v in directed_pairs:
        adj[u].append(v)
    for u in range(V):
        if u not in adj[u]:
            adj[u].append(u)

    # 2) Transition feasibility: pos(t)=u -> OR_{v in adj[u]} pos(t+1)=v
    for t in range(T):
        for u in range(V):
            lhs = eq_pos_const(t, u, b)
            succs = adj[u]
            rhs = eq_pos_const(t + 1, succs[0], b)
            for v in succs[1:]:
                rhs = rhs | eq_pos_const(t + 1, v, b)
            formulas.append(implies(lhs, rhs))

    # 3) Edge legality and consistency.
    for t in range(T):
        for u, v in directed_pairs:
            formulas.append(
                implies(
                    eq_pos_const(t, u, b) & eq_pos_const(t + 1, v, b),
                    Allowed(u, v),
                )
            )
    for u, v in directed_pairs:
        formulas.append(implies(Allowed(u, v), Active(u, v)))

    return formulas


def physics(
    T: int, V: int, directed_pairs: List[Tuple[int, int]] | None = None
) -> List[Expr]:
    """
    Compatibility wrapper; primary encoding is position-bit based.
    """
    return physics_bits(T=T, V=V, directed_pairs=directed_pairs)


def bob_physics(
    edges: List[Edge], V: int, directed_pairs: List[Tuple[int, int]] | None = None
) -> Tuple[List[Expr], Dict[Tuple[int, int], float]]:
    """
    Bob's view of edge traversability, with a closed-world assumption.

    Args:
        edges: list of Edge objects carrying Edge.state and optional Edge.weight:
            state 0 = no edge
            state 1 = edge exists but is not traversable
            state 2 = edge exists and is traversable
            weight = relative importance (None uses default SOFT_CLAUSE_WEIGHT)
        V: number of vertices (0..V-1).

    Returns:
        Tuple of:
          - Formulas asserting:
                state 2 -> Allowed(u,v) and Active(u,v)
                state 1 -> Active(u,v) and ¬Allowed(u,v)
                state 0 -> ¬Active(u,v) and ¬Allowed(u,v)
                Allowed(u,v) -> Active(u,v) for all u != v
          - Dict mapping (u,v) -> weight for edges with custom weights
    """
    formulas: List[Expr] = []
    edge_states = [[int(EdgeState.NO_EDGE) for _ in range(V)] for _ in range(V)]
    edge_weights: Dict[Tuple[int, int], float] = {}

    for edge in edges:
        u = edge.vertex1.id
        v = edge.vertex2.id
        if not (0 <= u < V and 0 <= v < V):
            raise ValueError("Edge vertex id out of range for V")
        if u == v:
            continue
        edge_states[u][v] = int(edge.state)
        if edge.weight is not None:
            edge_weights[(u, v)] = edge.weight

    pair_domain = directed_pairs or [
        (u, v) for u in range(V) for v in range(V) if u != v
    ]
    for u, v in pair_domain:
        state = edge_states[u][v]
        if state == int(EdgeState.TRAVERSABLE):
            formulas.append(Active(u, v))
            formulas.append(Allowed(u, v))
        elif state == int(EdgeState.BLOCKED):
            formulas.append(Active(u, v))
            formulas.append(~Allowed(u, v))
        else:
            formulas.append(~Active(u, v))
            formulas.append(~Allowed(u, v))
        formulas.append(implies(Allowed(u, v), Active(u, v)))

    return formulas, edge_weights


def alice_physics(start: int, goal: int, T: int, V: int) -> List[Expr]:
    """
    Alice's initial and goal conditions.


    Args:
        start: start vertex index.
        goal: goal vertex index.
        T: time horizon.
        V: number of vertices (unused but kept for symmetry / future use).

    Returns:
        Bit-level unit clauses enforcing:
          - pos(0) == start
          - pos(T) == goal
    """
    if T < 0:
        raise ValueError("T must be non-negative")
    if not (0 <= start < V):
        raise ValueError("start must be in [0, V-1]")
    if not (0 <= goal < V):
        raise ValueError("goal must be in [0, V-1]")

    b = num_bits(V)
    return force_pos_const(0, start, b) + force_pos_const(T, goal, b)


def ordered_symbols(
    T: int, V: int, directed_pairs: List[Tuple[int, int]] | None = None
) -> List[Expr]:
    """
    Deterministic variable ordering for Q columns:
      1) PosBit(t,k)
      2) Active(u,v), u!=v
      3) Allowed(u,v), u!=v
    """
    symbols: List[Expr] = []
    directed_pairs = directed_pairs or [
        (u, v) for u in range(V) for v in range(V) if u != v
    ]
    b = num_bits(V)

    for t in range(T + 1):
        for k in range(b):
            symbols.append(PosBit(t, k))

    for u, v in directed_pairs:
        symbols.append(Active(u, v))

    for u, v in directed_pairs:
        symbols.append(Allowed(u, v))

    return symbols


def assignment_from_u_vector(
    u_vector: List[int], T: int, V: int, symbols: List[Expr] | None = None
) -> Dict[Expr, bool]:
    """
    Decode a binary u-vector (length n) into an Expr->bool assignment map.

    Args:
        u_vector: Binary assignment values in the same order as `ordered_symbols(T, V)`.
        T: Time horizon used for symbol construction.
        V: Number of vertices used for symbol construction.
        symbols: Optional precomputed symbol list to avoid recomputation.

    Returns:
        Dict mapping each symbol Expr to a Python bool.
    """
    syms = symbols or ordered_symbols(T, V)
    if len(u_vector) != len(syms):
        raise ValueError(
            f"u_vector length ({len(u_vector)}) must equal num symbols ({len(syms)})"
        )
    return {sym: bool(int(val)) for sym, val in zip(syms, u_vector)}


def decode_path_from_assignment(assn: Dict[Expr, bool], T: int, V: int) -> List[int]:
    """
    Decode vertex ids from a symbol assignment using PosBit atoms.

    For each time t, reconstruct:
      path[t] = sum(2**k for k where PosBit(t,k) is True)

    With range restrictions active, each decoded value should be in [0, V-1].
    """
    b = num_bits(V)
    path: List[int] = []
    for t in range(T + 1):
        value = 0
        for k in range(b):
            if assn.get(PosBit(t, k), False):
                value |= 1 << k
        path.append(value)
    return path


def _flatten_cnf_clauses(formulas: List[Expr]) -> List[Expr]:
    clauses: List[Expr] = []
    for formula in formulas:
        clauses.extend(conjuncts(to_cnf(formula)))
    return clauses


def _clauses_to_q(clauses: List[Expr], symbols: List[Expr]) -> np.ndarray:
    """
    Encode CNF clauses as MatSat Q rows of shape (m, 2n):
      - first n columns for positive literals
      - next n columns for negative literals
    """
    n = len(symbols)
    q = np.zeros((len(clauses), 2 * n), dtype=np.int32)
    symbol_index = {sym: i for i, sym in enumerate(symbols)}

    for i, clause in enumerate(clauses):
        for lit in disjuncts(clause):
            if lit.op == "~":
                sym = lit.args[0]
                idx = symbol_index.get(sym)
                if idx is not None:
                    q[i, n + idx] = 1
            else:
                idx = symbol_index.get(lit)
                if idx is not None:
                    q[i, idx] = 1
    return q


def _is_exact_not_allowed_clause(clause: Expr) -> Tuple[bool, Tuple[int, int] | None]:
    """
    Return (True, (u, v)) iff clause is exactly a negated Allowed(u, v) literal.
    Return (False, None) otherwise.
    """
    if (
        clause.op == "~"
        and len(clause.args) == 1
        and clause.args[0].op == "Allowed"
        and len(clause.args[0].args) == 2
    ):
        u, v = clause.args[0].args
        return True, (int(u), int(v))
    return False, None


def _clause_weights_for_cnf(
    clauses: List[Expr],
    edge_weights: Dict[Tuple[int, int], float] | None = None,
    hard_clause_weight: float = DEFAULT_HARD_CLAUSE_WEIGHT,
) -> np.ndarray:
    """
    Clause-weight vector aligned with CNF row ordering.
    - Unit ~Allowed(u,v) clauses are soft, with weight from edge_weights if available.
    - All other clauses are hard (using hard_clause_weight).

    Args:
        clauses: List of CNF clauses
        edge_weights: Optional dict mapping (u,v) -> weight for soft clauses
        hard_clause_weight: Weight for hard clauses (default 1.0, but should be
                           num_parties + 1 to dominate soft clauses)
    """
    weights = np.full((len(clauses),), hard_clause_weight, dtype=np.float64)
    edge_weights = edge_weights or {}

    for i, clause in enumerate(clauses):
        is_not_allowed, edge_pair = _is_exact_not_allowed_clause(clause)
        if is_not_allowed and edge_pair is not None:
            # Use custom edge weight if provided, otherwise default
            weights[i] = edge_weights.get(edge_pair, SOFT_CLAUSE_WEIGHT)
    return weights


def build_physics_q(
    T: int,
    V: int,
    symbols: List[Expr] | None = None,
    directed_pairs: List[Tuple[int, int]] | None = None,
    print_cnf_clauses: bool = False,
    hard_clause_weight: float = DEFAULT_HARD_CLAUSE_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    syms = symbols or ordered_symbols(T, V, directed_pairs=directed_pairs)
    formulas = physics(T, V, directed_pairs=directed_pairs)
    cnf_clauses = _flatten_cnf_clauses(formulas)
    clause_weights = _clause_weights_for_cnf(
        cnf_clauses, hard_clause_weight=hard_clause_weight
    )
    if print_cnf_clauses:
        print("=== build_physics_q: Physics CNF clauses ===")
        clause_idx = 0
        for formula_idx, formula in enumerate(formulas):
            converted = conjuncts(to_cnf(formula))
            print(f"physics[{formula_idx}] = {formula}")
            for local_idx, clause in enumerate(converted):
                print(
                    f"  cnf[{clause_idx}] (from local {local_idx}, "
                    f"weight={clause_weights[clause_idx]}) = {clause}"
                )
                clause_idx += 1
        print(f"Total CNF clauses: {len(cnf_clauses)}")
    q = _clauses_to_q(cnf_clauses, syms)
    return q, clause_weights


# =============================================================================
# Internal helper functions (use build_*_q with PrivatePathInfo for public API)
# =============================================================================


def _build_bob_q(
    edges: List[Edge],
    T: int,
    V: int,
    symbols: List[Expr] | None = None,
    directed_pairs: List[Tuple[int, int]] | None = None,
    print_cnf_clauses: bool = False,
    hard_clause_weight: float = DEFAULT_HARD_CLAUSE_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper: Build Bob's Q matrix from his edge constraints.

    Use build_bob_q() with PrivatePathInfo for the validated public API.
    """
    syms = symbols or ordered_symbols(T, V, directed_pairs=directed_pairs)
    formulas, edge_weights = bob_physics(edges, V, directed_pairs=directed_pairs)
    cnf_clauses = _flatten_cnf_clauses(formulas)
    clause_weights = _clause_weights_for_cnf(
        cnf_clauses, edge_weights=edge_weights, hard_clause_weight=hard_clause_weight
    )
    if print_cnf_clauses:
        print("=== build_bob_q: Bob physics CNF clauses ===")
        clause_idx = 0
        for formula_idx, formula in enumerate(formulas):
            converted = conjuncts(to_cnf(formula))
            print(f"bob_physics[{formula_idx}] = {formula}")
            for local_idx, clause in enumerate(converted):
                print(
                    f"  cnf[{clause_idx}] (from local {local_idx}, "
                    f"weight={clause_weights[clause_idx]}) = {clause}"
                )
                clause_idx += 1
        print(f"Total CNF clauses: {len(cnf_clauses)}")
        if edge_weights:
            print(f"Custom edge weights: {edge_weights}")
    q = _clauses_to_q(cnf_clauses, syms)
    return q, clause_weights


def _build_alice_q(
    start: int,
    goal: int,
    T: int,
    V: int,
    symbols: List[Expr] | None = None,
    print_cnf_clauses: bool = False,
    hard_clause_weight: float = DEFAULT_HARD_CLAUSE_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper: Build Alice's Q matrix from start/goal constraints.

    Use build_alice_q() with PrivatePathInfo for the validated public API.
    """
    syms = symbols or ordered_symbols(T, V)
    formulas = alice_physics(start, goal, T, V)
    cnf_clauses = _flatten_cnf_clauses(formulas)
    clause_weights = _clause_weights_for_cnf(
        cnf_clauses, hard_clause_weight=hard_clause_weight
    )
    if print_cnf_clauses:
        print("=== build_alice_q: Alice physics CNF clauses ===")
        clause_idx = 0
        for formula_idx, formula in enumerate(formulas):
            converted = conjuncts(to_cnf(formula))
            print(f"alice_physics[{formula_idx}] = {formula}")
            for local_idx, clause in enumerate(converted):
                print(
                    f"  cnf[{clause_idx}] (from local {local_idx}, "
                    f"weight={clause_weights[clause_idx]}) = {clause}"
                )
                clause_idx += 1
        print(f"Total CNF clauses: {len(cnf_clauses)}")
    q = _clauses_to_q(cnf_clauses, syms)
    return q, clause_weights


def _build_q(
    start: int,
    goal: int,
    T: int,
    V: int,
    edges: List[Edge],
    use_edge_domain: bool = False,
) -> Tuple[np.ndarray, List[Expr], Dict[str, np.ndarray]]:
    """
    Internal helper: Build full MatSat Q by vertically concatenating components.

    Use build_q() with PrivatePathInfo for the validated public API.
    """
    directed_pairs = directed_pairs_from_edges(edges, V) if use_edge_domain else None
    syms = ordered_symbols(T, V, directed_pairs=directed_pairs)
    q_physics, w_physics = build_physics_q(
        T, V, symbols=syms, directed_pairs=directed_pairs
    )
    q_bob, w_bob = _build_bob_q(
        edges, T, V, symbols=syms, directed_pairs=directed_pairs
    )
    q_alice, w_alice = _build_alice_q(start, goal, T, V, symbols=syms)
    q_full = np.concatenate([q_physics, q_bob, q_alice], axis=0)
    w_full = np.concatenate([w_physics, w_bob, w_alice], axis=0)
    return (
        q_full,
        syms,
        {
            "physics": q_physics,
            "bob": q_bob,
            "alice": q_alice,
            "physics_weights": w_physics,
            "bob_weights": w_bob,
            "alice_weights": w_alice,
            "weights": w_full,
        },
    )


# =============================================================================
# Public API: PrivatePathInfo-based functions with validation
# =============================================================================


def build_alice_q(
    info: PrivatePathInfo,
    start: int,
    goal: int,
    symbols: List[Expr] | None = None,
    print_cnf_clauses: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Alice's Q matrix using shared config from PrivatePathInfo.

    Args:
        info: Shared path-finding configuration (T, V, edge domain)
        start: Alice's private start vertex
        goal: Alice's private goal vertex
        symbols: Optional pre-computed symbol list
        print_cnf_clauses: If True, print debug info

    Returns:
        Tuple of (Q matrix, clause weights array)

    Raises:
        AssertionError: If inputs are invalid
    """
    # Validate PrivatePathInfo
    assert info is not None, "info (PrivatePathInfo) is required"
    assert info.T >= 0, f"info.T must be >= 0, got {info.T}"
    assert info.V >= 1, f"info.V must be >= 1, got {info.V}"

    # Validate Alice's private inputs
    assert isinstance(start, int), f"start must be int, got {type(start)}"
    assert isinstance(goal, int), f"goal must be int, got {type(goal)}"
    assert 0 <= start < info.V, f"start must be in [0, {info.V-1}], got {start}"
    assert 0 <= goal < info.V, f"goal must be in [0, {info.V-1}], got {goal}"

    # Compute hard clause weight based on number of parties
    hard_weight = compute_hard_clause_weight(info.num_parties)

    return _build_alice_q(
        start=start,
        goal=goal,
        T=info.T,
        V=info.V,
        symbols=symbols,
        print_cnf_clauses=print_cnf_clauses,
        hard_clause_weight=hard_weight,
    )


def build_bob_q(
    info: PrivatePathInfo,
    edges: List[Edge],
    symbols: List[Expr] | None = None,
    print_cnf_clauses: bool = False,
    normalize_weights: bool = True,
    weight_budget: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Bob's Q matrix using shared config from PrivatePathInfo.

    Edge weights (if provided on Edge objects) are used for ~Allowed(u,v) clauses.
    This allows Bob to assign relative importance to different edges.

    Weights are automatically normalized to sum to weight_budget (default 1.0).
    Edges without explicit weights are assigned equal share of the budget.

    Args:
        info: Shared path-finding configuration (T, V, edge domain)
        edges: Bob's private edges with states (and optional weights)
        symbols: Optional pre-computed symbol list
        print_cnf_clauses: If True, print debug info
        normalize_weights: If True (default), normalize edge weights to sum to weight_budget
        weight_budget: Total weight budget for soft clauses (default 1.0)

    Returns:
        Tuple of (Q matrix, clause weights array)

    Raises:
        AssertionError: If inputs are invalid
    """
    # Validate PrivatePathInfo
    assert info is not None, "info (PrivatePathInfo) is required"
    assert info.T >= 0, f"info.T must be >= 0, got {info.T}"
    assert info.V >= 1, f"info.V must be >= 1, got {info.V}"

    # Validate Bob's private edges
    assert edges is not None, "edges list is required"
    assert isinstance(edges, list), f"edges must be a list, got {type(edges)}"
    for i, edge in enumerate(edges):
        assert isinstance(edge, Edge), f"edges[{i}] must be Edge, got {type(edge)}"
        assert (
            0 <= edge.vertex1.id < info.V
        ), f"edges[{i}].vertex1.id must be in [0, {info.V-1}], got {edge.vertex1.id}"
        assert (
            0 <= edge.vertex2.id < info.V
        ), f"edges[{i}].vertex2.id must be in [0, {info.V-1}], got {edge.vertex2.id}"
        assert edge.state in (
            EdgeState.TRAVERSABLE,
            EdgeState.BLOCKED,
            EdgeState.NO_EDGE,
        ), f"edges[{i}].state must be TRAVERSABLE, BLOCKED, or NO_EDGE, got {edge.state}"

    # Normalize edge weights to sum to weight_budget
    if normalize_weights and edges:
        edges = normalize_edge_weights(edges, total_budget=weight_budget)

    # Get directed pairs from edge domain if enabled
    directed_pairs = info.edge_domain_pairs() if info.use_edge_domain else None

    # Compute hard clause weight based on number of parties
    hard_weight = compute_hard_clause_weight(info.num_parties)

    return _build_bob_q(
        edges=edges,
        T=info.T,
        V=info.V,
        symbols=symbols,
        directed_pairs=directed_pairs,
        print_cnf_clauses=print_cnf_clauses,
        hard_clause_weight=hard_weight,
    )


def build_q(
    info: PrivatePathInfo,
    start: int,
    goal: int,
    edges: List[Edge],
    print_cnf_clauses: bool = False,
) -> Tuple[np.ndarray, List[Expr], Dict[str, np.ndarray]]:
    """
    Build full MatSat Q matrix using shared config from PrivatePathInfo.

    Combines physics, Bob's edge constraints, and Alice's start/goal constraints.

    Args:
        info: Shared path-finding configuration (T, V, edge domain)
        start: Alice's private start vertex
        goal: Alice's private goal vertex
        edges: Bob's private edges with states (and optional weights)
        print_cnf_clauses: If True, print debug info for all components

    Returns:
        Tuple of (full Q matrix, symbols list, component dict)

    Raises:
        AssertionError: If inputs are invalid
    """
    # Validate PrivatePathInfo
    assert info is not None, "info (PrivatePathInfo) is required"
    assert info.T >= 0, f"info.T must be >= 0, got {info.T}"
    assert info.V >= 1, f"info.V must be >= 1, got {info.V}"

    # Validate Alice's private inputs
    assert isinstance(start, int), f"start must be int, got {type(start)}"
    assert isinstance(goal, int), f"goal must be int, got {type(goal)}"
    assert 0 <= start < info.V, f"start must be in [0, {info.V-1}], got {start}"
    assert 0 <= goal < info.V, f"goal must be in [0, {info.V-1}], got {goal}"

    # Validate Bob's private edges
    assert edges is not None, "edges list is required"
    assert isinstance(edges, list), f"edges must be a list, got {type(edges)}"
    for i, edge in enumerate(edges):
        assert isinstance(edge, Edge), f"edges[{i}] must be Edge, got {type(edge)}"
        assert (
            0 <= edge.vertex1.id < info.V
        ), f"edges[{i}].vertex1.id must be in [0, {info.V-1}], got {edge.vertex1.id}"
        assert (
            0 <= edge.vertex2.id < info.V
        ), f"edges[{i}].vertex2.id must be in [0, {info.V-1}], got {edge.vertex2.id}"

    # Get directed pairs from edge domain if enabled
    directed_pairs = info.edge_domain_pairs() if info.use_edge_domain else None

    # Build symbol list
    syms = ordered_symbols(info.T, info.V, directed_pairs=directed_pairs)

    # Compute hard clause weight based on number of parties
    hard_weight = compute_hard_clause_weight(info.num_parties)

    # Build component Q matrices
    q_physics, w_physics = build_physics_q(
        info.T,
        info.V,
        symbols=syms,
        directed_pairs=directed_pairs,
        print_cnf_clauses=print_cnf_clauses,
        hard_clause_weight=hard_weight,
    )
    q_bob, w_bob = _build_bob_q(
        edges,
        info.T,
        info.V,
        symbols=syms,
        directed_pairs=directed_pairs,
        print_cnf_clauses=print_cnf_clauses,
        hard_clause_weight=hard_weight,
    )
    q_alice, w_alice = _build_alice_q(
        start,
        goal,
        info.T,
        info.V,
        symbols=syms,
        print_cnf_clauses=print_cnf_clauses,
        hard_clause_weight=hard_weight,
    )

    # Concatenate
    q_full = np.concatenate([q_physics, q_bob, q_alice], axis=0)
    w_full = np.concatenate([w_physics, w_bob, w_alice], axis=0)

    return (
        q_full,
        syms,
        {
            "physics": q_physics,
            "bob": q_bob,
            "alice": q_alice,
            "physics_weights": w_physics,
            "bob_weights": w_bob,
            "alice_weights": w_alice,
            "weights": w_full,
        },
    )

from utils4e import Expr
from logic4e import implies, equiv, to_cnf, conjuncts, disjuncts
from typing import List, Dict, Tuple
import numpy as np
from private_path_query_utils import EdgeState, Edge


HARD_CLAUSE_WEIGHT = 1.0
SOFT_CLAUSE_WEIGHT = 0.1


# HARD EXPRESSIONS: Must be satisfied


def At(t: int, v: int) -> Expr:
    return Expr("At", t, v)


def Move(t: int, u: int, v: int) -> Expr:
    return Expr("Move", t, u, v)


def Wait(t: int, v: int) -> Expr:
    return Expr("Wait", t, v)


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


def physics(
    T: int, V: int, directed_pairs: List[Tuple[int, int]] | None = None
) -> List[Expr]:
    """
    Propositional dynamics for a single agent moving on a directed graph.

    Args:
        T: time horizon (number of steps).
        V: number of vertices (assumed to be 0..V-1).

    Returns:
        List of logical formulas encoding:
          - exactly-one At(t, ·) for each t
          - exactly-one action per t: either Move(t, u, v) over all u!=v, or Wait(t, v)
          - transition preconditions:
                Move(t,u,v) -> At(t,u)
                Wait(t,v) -> At(t,v)
                Move(t,u,v) -> Allowed(u,v)
          - edge consistency:
                Allowed(u,v) -> Active(u,v)
          - successor-state axiom for each vertex/time:
                At(t+1,v) <-> (Wait(t,v) OR OR_u Move(t,u,v))
    """
    formulas: List[Expr] = []
    directed_pairs = directed_pairs or [
        (u, v) for u in range(V) for v in range(V) if u != v
    ]

    # 1) Exactly-one position at each time step
    for t in range(T + 1):
        at_literals = [At(t, v) for v in range(V)]
        formulas.extend(_exactly_one(at_literals))

    # 2) Exactly-one action per time step (move or wait-at-vertex)
    for t in range(T):
        action_literals = [Move(t, u, v) for (u, v) in directed_pairs] + [
            Wait(t, v) for v in range(V)
        ]
        formulas.extend(_exactly_one(action_literals))

    # 3) Local transition preconditions for each possible move.
    for t in range(T):
        for u, v in directed_pairs:
            m = Move(t, u, v)
            # If we move u->v at time t, we must be at u at time t
            formulas.append(implies(m, At(t, u)))
            # ...and the edge must be allowed for traversal
            formulas.append(implies(m, Allowed(u, v)))
            # Allowed traversal implies edge exists.
            formulas.append(implies(Allowed(u, v), Active(u, v)))

    # 4) Wait preconditions
    for t in range(T):
        for v in range(V):
            w = Wait(t, v)
            formulas.append(implies(w, At(t, v)))

    # 5) Successor-state axiom:
    #    At(t+1,v) <-> (Wait(t,v) OR incoming Move(t,*,v))
    for t in range(T):
        for v in range(V):
            incoming_moves = [Move(t, u, w) for (u, w) in directed_pairs if w == v]
            disj = Wait(t, v)
            for m in incoming_moves:
                disj = disj | m
            formulas.append(equiv(At(t + 1, v), disj))

    return formulas


def bob_physics(
    edges: List[Edge], V: int, directed_pairs: List[Tuple[int, int]] | None = None
) -> List[Expr]:
    """
    Bob's view of edge traversability, with a closed-world assumption.

    Args:
        edges: list of Edge objects carrying Edge.state:
            0 = no edge
            1 = edge exists but is not traversable
            2 = edge exists and is traversable
        V: number of vertices (0..V-1).

    Returns:
        Formulas asserting:
          - Edge-state mapping:
                state 2 -> Allowed(u,v) and Active(u,v)
                state 1 -> Active(u,v) and ¬Allowed(u,v)
                state 0 -> ¬Active(u,v) and ¬Allowed(u,v)
          - Active(u,v): edge existence (state in {1,2})
          - Allowed(u,v): traversal permission (state == 2)
          - Allowed(u,v) -> Active(u,v) for all u != v
    """
    formulas: List[Expr] = []
    edge_states = [[int(EdgeState.NO_EDGE) for _ in range(V)] for _ in range(V)]
    for edge in edges:
        u = edge.vertex1.id
        v = edge.vertex2.id
        if not (0 <= u < V and 0 <= v < V):
            raise ValueError("Edge vertex id out of range for V")
        if u == v:
            continue
        edge_states[u][v] = int(edge.state)

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

    return formulas


def alice_physics(start: int, goal: int, T: int, V: int) -> List[Expr]:
    """
    Alice's initial and goal conditions.


    Args:
        start: start vertex index.
        goal: goal vertex index.
        T: time horizon.
        V: number of vertices (unused but kept for symmetry / future use).

    Returns:
        Formulas enforcing:
          - At(0, start)
          - At(T, goal)  (goal must be true exactly at time T)
    """
    if T < 0:
        raise ValueError("T must be non-negative")
    if not (0 <= start < V):
        raise ValueError("start must be in [0, V-1]")
    if not (0 <= goal < V):
        raise ValueError("goal must be in [0, V-1]")

    formulas: List[Expr] = []
    formulas.append(At(0, start))
    formulas.append(At(T, goal))

    return formulas


def ordered_symbols(
    T: int, V: int, directed_pairs: List[Tuple[int, int]] | None = None
) -> List[Expr]:
    """
    Deterministic variable ordering for Q columns:
      1) At(t,v)
      2) Move(t,u,v), u!=v
      3) Wait(t,v)
      4) Active(u,v), u!=v
      5) Allowed(u,v), u!=v
    """
    symbols: List[Expr] = []
    directed_pairs = directed_pairs or [
        (u, v) for u in range(V) for v in range(V) if u != v
    ]

    for t in range(T + 1):
        for v in range(V):
            symbols.append(At(t, v))

    for t in range(T):
        for u, v in directed_pairs:
            symbols.append(Move(t, u, v))

    for t in range(T):
        for v in range(V):
            symbols.append(Wait(t, v))

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


def _is_exact_not_allowed_clause(clause: Expr) -> bool:
    """
    Return True iff clause is exactly a negated Allowed(u, v) literal.
    """
    return (
        clause.op == "~"
        and len(clause.args) == 1
        and clause.args[0].op == "Allowed"
        and len(clause.args[0].args) == 2
    )


def _clause_weights_for_cnf(clauses: List[Expr]) -> np.ndarray:
    """
    Clause-weight vector aligned with CNF row ordering.
    - Unit ~Allowed(u,v) clauses are soft.
    - All other clauses are hard.
    """
    weights = np.full((len(clauses),), HARD_CLAUSE_WEIGHT, dtype=np.float64)
    for i, clause in enumerate(clauses):
        if _is_exact_not_allowed_clause(clause):
            weights[i] = SOFT_CLAUSE_WEIGHT
    return weights


def build_physics_q(
    T: int,
    V: int,
    symbols: List[Expr] | None = None,
    directed_pairs: List[Tuple[int, int]] | None = None,
    print_cnf_clauses: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    syms = symbols or ordered_symbols(T, V, directed_pairs=directed_pairs)
    formulas = physics(T, V, directed_pairs=directed_pairs)
    cnf_clauses = _flatten_cnf_clauses(formulas)
    clause_weights = _clause_weights_for_cnf(cnf_clauses)
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


def build_bob_q(
    edges: List[Edge],
    T: int,
    V: int,
    symbols: List[Expr] | None = None,
    directed_pairs: List[Tuple[int, int]] | None = None,
    print_cnf_clauses: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    syms = symbols or ordered_symbols(T, V, directed_pairs=directed_pairs)
    formulas = bob_physics(edges, V, directed_pairs=directed_pairs)
    cnf_clauses = _flatten_cnf_clauses(formulas)
    clause_weights = _clause_weights_for_cnf(cnf_clauses)
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
    q = _clauses_to_q(cnf_clauses, syms)
    return q, clause_weights


def build_alice_q(
    start: int,
    goal: int,
    T: int,
    V: int,
    symbols: List[Expr] | None = None,
    print_cnf_clauses: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    syms = symbols or ordered_symbols(T, V)
    formulas = alice_physics(start, goal, T, V)
    cnf_clauses = _flatten_cnf_clauses(formulas)
    clause_weights = _clause_weights_for_cnf(cnf_clauses)
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


def build_q(
    start: int,
    goal: int,
    T: int,
    V: int,
    edges: List[Edge],
    use_edge_domain: bool = False,
) -> Tuple[np.ndarray, List[Expr], Dict[str, np.ndarray]]:
    """
    Build full MatSat Q by vertically concatenating:
      1) physics clauses
      2) bob edge-state clauses
      3) alice start/goal clauses
    """
    directed_pairs = directed_pairs_from_edges(edges, V) if use_edge_domain else None
    syms = ordered_symbols(T, V, directed_pairs=directed_pairs)
    q_physics, w_physics = build_physics_q(
        T, V, symbols=syms, directed_pairs=directed_pairs
    )
    q_bob, w_bob = build_bob_q(edges, T, V, symbols=syms, directed_pairs=directed_pairs)
    q_alice, w_alice = build_alice_q(start, goal, T, V, symbols=syms)
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

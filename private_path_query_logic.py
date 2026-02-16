from utils4e import Expr
from logic4e import implies, equiv, to_cnf, conjuncts, disjuncts
from typing import List, Dict, Tuple
import numpy as np
from private_path_query_utils import EdgeState, Edge


def At(t: int, v: int) -> Expr:
    return Expr("At", t, v)


def Move(t: int, u: int, v: int) -> Expr:
    return Expr("Move", t, u, v)


def Wait(t: int, v: int) -> Expr:
    return Expr("Wait", t, v)


def Active(u: int, v: int) -> Expr:
    return Expr("Active", u, v)


def Allowed(u: int, v: int) -> Expr:
    return Expr("Allowed", u, v)


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


def physics(T: int, V: int) -> List[Expr]:
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
    directed_pairs = [(u, v) for u in range(V) for v in range(V) if u != v]

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
            incoming_moves = [Move(t, u, v) for u in range(V) if u != v]
            disj = Wait(t, v)
            for m in incoming_moves:
                disj = disj | m
            formulas.append(equiv(At(t + 1, v), disj))

    return formulas


def bob_physics(edges: List[Edge], V: int) -> List[Expr]:
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

    for u in range(V):
        for v in range(V):
            if u == v:
                continue
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


def ordered_symbols(T: int, V: int) -> List[Expr]:
    """
    Deterministic variable ordering for Q columns:
      1) At(t,v)
      2) Move(t,u,v), u!=v
      3) Wait(t,v)
      4) Active(u,v), u!=v
      5) Allowed(u,v), u!=v
    """
    symbols: List[Expr] = []
    directed_pairs = [(u, v) for u in range(V) for v in range(V) if u != v]

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


def build_physics_q(T: int, V: int, symbols: List[Expr] | None = None) -> np.ndarray:
    syms = symbols or ordered_symbols(T, V)
    return _clauses_to_q(_flatten_cnf_clauses(physics(T, V)), syms)


def build_bob_q(
    edges: List[Edge], T: int, V: int, symbols: List[Expr] | None = None
) -> np.ndarray:
    syms = symbols or ordered_symbols(T, V)
    return _clauses_to_q(_flatten_cnf_clauses(bob_physics(edges, V)), syms)


def build_alice_q(
    start: int, goal: int, T: int, V: int, symbols: List[Expr] | None = None
) -> np.ndarray:
    syms = symbols or ordered_symbols(T, V)
    return _clauses_to_q(_flatten_cnf_clauses(alice_physics(start, goal, T, V)), syms)


def build_q(
    start: int, goal: int, T: int, V: int, edges: List[Edge]
) -> Tuple[np.ndarray, List[Expr], Dict[str, np.ndarray]]:
    """
    Build full MatSat Q by vertically concatenating:
      1) physics clauses
      2) bob edge-state clauses
      3) alice start/goal clauses
    """
    syms = ordered_symbols(T, V)
    q_physics = build_physics_q(T, V, symbols=syms)
    q_bob = build_bob_q(edges, T, V, symbols=syms)
    q_alice = build_alice_q(start, goal, T, V, symbols=syms)
    q_full = np.concatenate([q_physics, q_bob, q_alice], axis=0)
    return q_full, syms, {"physics": q_physics, "bob": q_bob, "alice": q_alice}

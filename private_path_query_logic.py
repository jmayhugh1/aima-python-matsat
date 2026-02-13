from utils4e import Expr
from logic4e import implies, equiv, new_disjunction
from typing import List, Tuple


def At(t: int, v: int) -> Expr:
    return Expr("At", t, v)


def Move(t: int, u: int, v: int) -> Expr:
    return Expr("Move", t, u, v)


def Active(u: int, v: int) -> Expr:
    return Expr("Active", u, v)


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


def physics(E_dir: List[Tuple[int, int]], T: int, V: int) -> List[Expr]:
    """
    Propositional dynamics for a single agent moving on a directed graph.

    Args:
        E_dir: list of directed edges (u, v) that exist in the public topology.
        T: time horizon (number of steps).
        V: number of vertices (assumed to be 0..V-1).

    Returns:
        List of logical formulas encoding:
          - exactly-one At(t, ·) for each t
          - exactly-one Move(t, ·, ·) over (u,v) in E_dir for each t
          - transition preconditions/effects:
                Move(t,u,v) -> At(t,u)
                Move(t,u,v) -> At(t+1,v)
                Move(t,u,v) -> Active(u,v)
          - optional "frame" direction:
                At(t+1,v) -> OR_u Move(t,u,v) for incoming edges
    """
    formulas: List[Expr] = []

    # 1) Exactly-one position at each time step
    for t in range(T + 1):
        at_literals = [At(t, v) for v in range(V)]
        formulas.extend(_exactly_one(at_literals))

    # 2) Exactly-one move per time step (sequential plan)
    for t in range(T):
        move_literals = [Move(t, u, v) for (u, v) in E_dir]
        formulas.extend(_exactly_one(move_literals))

    # 3) Local transition dynamics for each possible move (only over real edges)
    for t in range(T):
        for u, v in E_dir:
            m = Move(t, u, v)
            # If we move u->v at time t, we must be at u at time t
            formulas.append(implies(m, At(t, u)))
            # ...and at v at time t+1
            formulas.append(implies(m, At(t + 1, v)))
            # ...and the edge must be active
            formulas.append(implies(m, Active(u, v)))

    # 4) Optional "frame" direction: At(t+1,v) must be justified by some incoming move
    for t in range(T):
        for v in range(V):
            incoming_moves = [Move(t, u, v2) for (u, v2) in E_dir if v2 == v]
            if not incoming_moves:
                continue
            disj = incoming_moves[0]
            for m in incoming_moves[1:]:
                disj = disj | m
            formulas.append(implies(At(t + 1, v), disj))

    return formulas


def bob_physics(allowed_edges: List[Tuple[int, int]], V: int) -> List[Expr]:
    """
    Bob's view of which edges are allowed (active), with a closed-world assumption.

    Args:
        allowed_edges: list of *undirected* edges Bob allows to be used.
                       For each (u, v) we treat both (u, v) and (v, u) as allowed.
        V: number of vertices (0..V-1).

    Returns:
        Formulas asserting:
          - Active(u,v) and Active(v,u) for allowed edges
          - ¬Active(u,v) for all other ordered pairs u != v
    """
    formulas: List[Expr] = []
    allowed_set = set()
    for u, v in allowed_edges:
        # store both directions as allowed
        allowed_set.add((u, v))
        allowed_set.add((v, u))
        formulas.append(Active(u, v))
        formulas.append(Active(v, u))

    # Closed-world: everything not in allowed_set is explicitly inactive
    for u in range(V):
        for v in range(V):
            if u == v:
                continue
            if (u, v) not in allowed_set:
                formulas.append(~Active(u, v))

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
        If you want "reach by time T", you could instead add OR_t At(t, goal).
    """
    formulas: List[Expr] = []
    formulas.append(At(0, start))
    
    or_goals = []
    for t in range(T):
        or_goals.append(At(t, goal))
    formulas.append(new_disjunction(or_goals))

    return formulas

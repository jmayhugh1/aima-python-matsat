from typing import *
from utils4e import Expr, expr
from logic4e import to_cnf, conjuncts, prop_symbols, dpll_satisfiable
import numpy as np
import cppimport

# ---------- helpers: build instance matrices ----------


def _symbols_order(formula: Expr) -> List[Expr]:
    """Deterministic symbol order [x1, x2, ..., xn] (sorted by strng name)."""
    cnf = to_cnf(formula)
    return sorted(list(prop_symbols(cnf)), key=str)


def _clause_literals(cl: Expr) -> List[Expr]:
    s = str(to_cnf(cl))
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return [expr(tok) for tok in s.split(" | ")]


def _build_Q1_Q2(formula: Expr) -> Tuple[np.ndarray, np.ndarray, List[Expr]]:
    """
    Returns Q1 (pos literals), Q2 (neg literals) see https://arxiv.org/abs/2108.06481, and ordered symbols.
    Q1, Q2 shapes: (m, n) where m=num clauses, n=num vars.
    """
    cnf = to_cnf(formula)
    clauses = conjuncts(cnf)
    syms = _symbols_order(cnf)
    n, m = len(syms), len(clauses)
    Q1 = np.zeros((m, n), dtype=np.float64)
    Q2 = np.zeros((m, n), dtype=np.float64)
    for i, cl in enumerate(clauses):
        lits = _clause_literals(cl)
        for j, s in enumerate(syms):
            if s in lits:
                Q1[i, j] = 1.0
            elif expr("~" + str(s)) in lits:
                Q2[i, j] = 1.0
    return Q1, Q2, syms


# ---------- core: MatSat solver returning a dict ----------


def mat_sat(
    formula: Expr,
    ell: float = 1.0,
    max_try: int = 8,
    max_itr: int = 200,
    beta: float = 0.25,
    seed: int = 0,
) -> Dict[str, bool] | bool:  # i picked random numbers for these
    """
    Solve SAT via MatSat cost minimization.
    Returns: {symbol_name: bool, ...} if satisfiable, else {}.
    """
    Q1, Q2, syms = _build_Q1_Q2(formula)
    n = len(syms)
    if n == 0:
        return {}

    rng = np.random.default_rng(seed)
    ones_n = np.ones(n, dtype=np.float64)
    A = Q1 - Q2  # (m, n)
    Q2_ones = Q2 @ ones_n  # (m,)

    def _min1(x: np.ndarray) -> np.ndarray:
        return np.minimum(1.0, x)

    def _J(u: np.ndarray) -> float:
        # Jsat = sum(1 - min1(Q u_d)) + (ell/2)||u ⊙ (1-u)||^2
        c = Q2_ones + (A @ u)  # (m,)
        clause_pen = np.sum(1.0 - _min1(c))
        bin_term = u * (1.0 - u)
        return clause_pen + 0.5 * ell * float(np.dot(bin_term, bin_term))

    def _grad(u: np.ndarray) -> np.ndarray:
        # ∇J = (Q2 - Q1)^T * (Q u_d)<1 + ell * (u ⊙ (1-u) ⊙ (1-2u))
        c = Q2_ones + (A @ u)  # (m,)
        mask = (c < 1.0).astype(np.float64)  # indicator of unsatisfied clauses
        term1 = (Q2 - Q1).T @ mask  # (n,)
        d = u * (1.0 - u) * (1.0 - 2.0 * u)  # (n,)
        return term1 + ell * d

    def _unsat_count(u_bin: np.ndarray) -> int:
        # number of unsatisfied clauses under binary u
        c_b = Q2_ones + (A @ u_bin)
        return int(np.sum(1.0 - _min1(c_b)))

    # random initial relaxed assignment
    u = rng.uniform(0.0, 1.0, size=n)

    best_u_bin = None
    best_err = np.inf

    for _ in range(max_try):
        for _ in range(max_itr):
            J = _J(u)
            g = _grad(u)
            gnorm2 = float(np.dot(g, g))
            if gnorm2 == 0.0:
                break
            # adaptive step size alpha = J / ||∇J||^2
            u -= (J / gnorm2) * g
            np.clip(u, 0.0, 1.0, out=u)

            u_bin = (u >= 0.5).astype(np.float64)
            err = _unsat_count(u_bin)
            if err < best_err:
                best_err = err
                best_u_bin = u_bin.copy()
                if err == 0:
                    # Found satisfying assignment → return mapping
                    return {str(sym): bool(val) for sym, val in zip(syms, best_u_bin)}

        # randomized restart / noise mixing
        u = (1.0 - beta) * u + beta * rng.uniform(0.0, 1.0, size=n)

    # No satisfying assignment found
    return (
        False  # could not satisfy
        if best_err != 0
        else {str(sym): bool(val) for sym, val in zip(syms, best_u_bin)}
    )


def mat_sat_cpp(formula : Expr) -> dict[Expr, bool] | None:
    if type(formula) == Expr:
        formula = to_cnf(formula)
    if type(formula) != str:
        formula = str(formula)
        if len(formula) >= 2 and formula[0] == '(' and formula[-1] == ')':
            formula = formula[1:-1]
    m = cppimport.imp("matsat")
    assignment = m.mat_sat(formula, max_itr = 100)
    return {expr(key) : val for key, val in assignment.items()} if assignment else None


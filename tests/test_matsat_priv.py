import pytest
import sys
import os

# Add parent directory to path so we can import logic4e
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mat_sat import *
from utils4e import expr
from utils import Expr
from logic4e import to_cnf, pl_true
import asyncio
import time


# run with pytest -vv -s --durations=0
def _mk_cnf_strings(var_names):
    """
    Build three satisfiable CNFs using the given variable list.
    All-True assignment satisfies everything (by construction).
    Returns (s1, s2, s3) as strings suitable for expr(...).
    """
    n = len(var_names)
    v = var_names

    # CNF 1: many 2-clauses (vi | v(i+1)), plus a few (vi | v(i+2)) to add spread
    c1 = []
    for i in range(n):
        c1.append(f"({v[i]} | {v[(i+1) % n]})")
    for i in range(0, n, max(3, n // 20)):
        c1.append(f"({v[i]} | {v[(i+2) % n]})")
    s1 = " & ".join(c1)

    # CNF 2: 3-clauses with one negated literal: (~vi | v(i+1) | v(i+2))
    c2 = []
    for i in range(0, n, 2):
        c2.append(f"(~{v[i]} | {v[(i+1) % n]} | {v[(i+2) % n]})")
    s2 = " & ".join(c2)

    # CNF 3: mix of 3-clauses and 2-clauses, still trivially true when all vars True
    c3 = []
    for i in range(0, n, 3):
        c3.append(f"({v[i]} | {v[(i+3) % n]} | {v[(i+5) % n]})")
    for i in range(1, n, 5):
        c3.append(f"({v[i]} | ~{v[(i+2) % n]})")
    s3 = " & ".join(c3)

    return s1, s2, s3


def _build_exprs(N):
    names = _excel_var_names(N)
    s1, s2, s3 = _mk_cnf_strings(names)
    return expr(s1), expr(s2), expr(s3)


def verifier(assignment: dict[str, bool], f: Expr | str):
    """verify that this is an adequate assignment to the formula"""
    # Convert string to Expr if needed
    if isinstance(f, str):
        formula = expr(f)
    else:
        formula = f

    # Check if assignment is empty or None
    if not assignment:
        return False

    # Try multiple approaches to verify the assignment
    # First, try with the original formula
    result = pl_true(formula, assignment)
    if result is True:
        return True

    # If pl_true returns None or False, try with CNF version
    # This handles cases where the solver worked on CNF but we're verifying original
    try:
        cnf_formula = to_cnf(formula)
        cnf_result = pl_true(cnf_formula, assignment)
        if cnf_result is True:
            return True
    except:
        pass

    # If both return None, it means the assignment is incomplete
    # For verification purposes, we consider this as False (not satisfying)
    return False


def combine_formulas(formula_list):
    """Helper to combine a list of formulas with &."""
    result = formula_list[0]
    for f in formula_list[1:]:
        result = result & f
    return result


def run_mpspdz_test(formulas, protocol="shamir", debug=True, print_result=True):
    """Helper function to run mat_sat_mpspdz and verify results."""
    combined_formula = combine_formulas(formulas)
    print("combined formula is", combined_formula)

    result = mat_sat_mpspdz(formulas, protocol=protocol, debug=debug)
    assert verifier(result, combined_formula), "Result does not satisfy the formula"

    if print_result:
        print("the result is", result)  # visible with `pytest -s`

    assert result is not None, "Result should not be None"
    return result


def run_timed_mpspdz_test(formulas, protocol="shamir", debug=True, test_name=""):
    """Helper function to run timed mat_sat_mpspdz tests."""
    combined_formula = combine_formulas(formulas)
    print("combined formula is", combined_formula)

    start = time.time()
    result = mat_sat_mpspdz(formulas, protocol=protocol, debug=debug)
    duration = time.time() - start

    assert verifier(result, combined_formula), "Result does not satisfy the formula"
    print("the result is", result)  # visible with `pytest -s`

    if test_name:
        print(f"{test_name} runtime: {duration:.3f}s")

    assert result is not None, "Result should not be None"
    return result


def test_two_party_matsat():
    f1 = expr("A | B")
    f2 = expr("~A | C")
    run_timed_mpspdz_test([f1, f2])


def test_timed_shamir_matsat_mpspdz_simple():
    f1 = expr("A | B | C")
    f2 = expr("A | ~B")
    f3 = expr("~A | C")
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir simple")


def test_timed_shamir_matsat_mpspdz_complex():
    f1 = expr("(A | B | C) & (D | E)")
    f2 = expr("(~A | C | D)")
    f3 = expr("(~C | E) & (B | ~D)")
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir complex")


def test_timed_shamir_matsat_mpspdz_8vars():
    # Three expressions; each may contain multiple clauses combined with &
    f1 = expr("(A | B | C) & (D | E)")
    f2 = expr("(~A | C | F) & (G | ~D)")
    f3 = expr("(~C | E | H) & (B | ~G | D)")
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_10vars():
    # Three expressions; each can combine multiple clauses with &
    f1 = expr("(A | B | C) & (D | E) & (F | G)")
    f2 = expr("(~A | C | H) & (I | ~D) & (E | F)")
    f3 = expr("(~C | E | J) & (B | ~G | D) & (H | I)")
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_12vars():
    # Three expressions; each may include multiple CNF clauses combined with &
    f1 = expr("(A | B | C) & (D | E) & (F | G) & (H | I)")
    f2 = expr("(~A | C | H) & (I | ~D) & (E | F) & (G | J)")
    f3 = expr("(~C | E | L) & (B | ~G | D) & (H | I | K) & (J | K | L)")
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_15vars():
    # Three expressions; each combines multiple CNF clauses with &
    f1 = expr("(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K)")
    f2 = expr("(~A | C | L) & (I | ~D) & (E | F) & (G | J) & (K | M)")
    f3 = expr("(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O)")
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_20vars():
    # Three expressions; each combines multiple CNF-style clauses with &
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & (L | M) & (N | O) & (P | Q) & (R | S) & (T | A)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & (N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & (P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J)"
    )
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_25vars():
    # Three expressions; each combines multiple CNF-style clauses with &
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & (V | W) & (X | Y)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | A)"
    )
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_30vars():
    # Variables: A, B, C, D, E, F, G, H, I, J,
    #            K, L, M, N, O, P, Q, R, S, T,
    #            U, V, W, X, Y, Z, AA, AB, AC, AD

    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | A)"
    )

    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | E)"
    )

    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | A | B) & (Q | R | S)"
    )
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_35vars():
    # Variables: A–Z, AA, AB, AC, AD, AE, AF, AG, AH, AI

    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y)"
    )

    f2 = expr(
        "(Z | AA) & (AB | AC) & (AD | AE) & (AF | AG) & (AH | AI) & "
        "(A | D | G) & (H | K | N) & (Q | T | W) & (Y | Z | AB) & "
        "(AC | AD | AE) & (AF | AG | AH) & (AI | A)"
    )

    f3 = expr(
        "(B | E | H) & (I | L | O) & (P | S | V) & (W | X | Y) & "
        "(Z | AA | AB) & (AC | AD) & (AE | AF) & (AG | AH) & "
        "(AI | B) & (C | F | I) & (J | M | P)"
    )
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_40vars():
    # Variables: A–Z, AA, AB, AC, AD, AE, AF, AG, AH, AI, AJ, AK, AL, AM, AN

    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | A)"
    )

    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & (Q | R | S)"
    )

    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | A) & (Q | T | W) & (Y | AB | AE) & (AC | AD | AN)"
    )
    run_timed_mpspdz_test([f1, f2, f3])


def test_timed_shamir_matsat_mpspdz_45vars():
    # Variables: A–Z (26), AA–AS (19) = 45 total
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS)"
    )

    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C)"
    )

    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | A | Q)"
    )

    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 45-vars")


def test_timed_shamir_matsat_mpspdz_50vars():
    # Variables: A–Z (26), AA–AX (24) = 50 total
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | A)"
    )

    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & (AX | F)"
    )

    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX)"
    )

    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 50-vars")


def test_timed_shamir_matsat_mpspdz_55vars():
    # Variables: A–Z (26), AA–BC (29) = 55 total
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | AY) & "
        "(AZ | BA) & (BB | BC) & (A | AH)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & "
        "(AX | AY | F) & (AZ | BA | G) & (BB | BC | H)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX) & (AY | AZ | BA) & (BB | BC | A)"
    )
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 55-vars")


def test_timed_shamir_matsat_mpspdz_60vars():
    # Variables: A–Z (26), AA–BH (34) = 60 total
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | AY) & "
        "(AZ | BA) & (BB | BC) & (BD | BE) & (BF | BG) & (BH | A)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & "
        "(AX | AY | F) & (AZ | BA | G) & (BB | BC | H) & (BD | BE | I) & "
        "(BF | BG | J) & (BH | K | L)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX) & (AY | AZ | BA) & (BB | BC | BD) & (BE | BF | BG) & "
        "(BH | A | Q)"
    )
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 60-vars")


def test_timed_shamir_matsat_mpspdz_65vars():
    # Variables: A–Z (26), AA–BH (34), BI–BM (5) = 65 total
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | AY) & "
        "(AZ | BA) & (BB | BC) & (BD | BE) & (BF | BG) & (BH | A) & "
        "(BI | BJ) & (BK | BL) & (BM | A)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & "
        "(AX | AY | F) & (AZ | BA | G) & (BB | BC | H) & (BD | BE | I) & "
        "(BF | BG | J) & (BH | K | L) & (BI | BJ | M) & (BK | BL | N) & (BM | O | P)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX) & (AY | AZ | BA) & (BB | BC | BD) & (BE | BF | BG) & "
        "(BH | BI | BJ) & (BK | BL | BM) & (A | Q | T)"
    )
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 65-vars")


def test_timed_shamir_matsat_mpspdz_75vars():
    # Variables: A–Z (26), AA–BH (34), BI–BW (20) = 80, but we need 75 total → up to BR (15 new)
    # New symbols beyond BH: BI, BJ, BK, BL, BM, BN, BO, BP, BQ, BR
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | AY) & "
        "(AZ | BA) & (BB | BC) & (BD | BE) & (BF | BG) & (BH | A) & "
        "(BI | BJ) & (BK | BL) & (BM | BN) & (BO | BP) & (BQ | BR)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & "
        "(AX | AY | F) & (AZ | BA | G) & (BB | BC | H) & (BD | BE | I) & "
        "(BF | BG | J) & (BH | K | L) & (BI | BJ | M) & (BK | BL | N) & "
        "(BM | BN | O) & (BO | BP | P) & (BQ | BR | Q)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX) & (AY | AZ | BA) & (BB | BC | BD) & (BE | BF | BG) & "
        "(BH | BI | BJ) & (BK | BL | BM) & (BN | BO | BP) & (BQ | BR | A)"
    )
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 75-vars")


def test_timed_shamir_matsat_mpspdz_85vars():
    # Variables: A–Z (26), AA–BH (34), BI–BW (20), BX–BZ (3), CA–CC (3) = 86, but we need 85 → up to CB
    # New symbols beyond BR used above: BS, BT, BU, BV, BW, BX, BY, BZ, CA, CB (10 new over 75)
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | AY) & "
        "(AZ | BA) & (BB | BC) & (BD | BE) & (BF | BG) & (BH | A) & "
        "(BI | BJ) & (BK | BL) & (BM | BN) & (BO | BP) & (BQ | BR) & "
        "(BS | BT) & (BU | BV) & (BW | BX) & (BY | BZ) & (CA | CB)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & "
        "(AX | AY | F) & (AZ | BA | G) & (BB | BC | H) & (BD | BE | I) & "
        "(BF | BG | J) & (BH | K | L) & (BI | BJ | M) & (BK | BL | N) & "
        "(BM | BN | O) & (BO | BP | P) & (BQ | BR | Q) & (BS | BT | R) & "
        "(BU | BV | S) & (BW | BX | T) & (BY | BZ | U) & (CA | CB | V)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX) & (AY | AZ | BA) & (BB | BC | BD) & (BE | BF | BG) & "
        "(BH | BI | BJ) & (BK | BL | BM) & (BN | BO | BP) & (BQ | BR | BS) & "
        "(BT | BU | BV) & (BW | BX | BY) & (BZ | CA | CB)"
    )
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 85-vars")


def test_timed_shamir_matsat_mpspdz_95vars():
    # Build on 85: add CC, CD, CE, CF, CG, CH, CI, CJ, CK, CL (10 new) = 95 total
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | AY) & "
        "(AZ | BA) & (BB | BC) & (BD | BE) & (BF | BG) & (BH | A) & "
        "(BI | BJ) & (BK | BL) & (BM | BN) & (BO | BP) & (BQ | BR) & "
        "(BS | BT) & (BU | BV) & (BW | BX) & (BY | BZ) & (CA | CB) & "
        "(CC | CD) & (CE | CF) & (CG | CH) & (CI | CJ) & (CK | CL)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & "
        "(AX | AY | F) & (AZ | BA | G) & (BB | BC | H) & (BD | BE | I) & "
        "(BF | BG | J) & (BH | K | L) & (BI | BJ | M) & (BK | BL | N) & "
        "(BM | BN | O) & (BO | BP | P) & (BQ | BR | Q) & (BS | BT | R) & "
        "(BU | BV | S) & (BW | BX | T) & (BY | BZ | U) & (CA | CB | V) & "
        "(CC | CD | W) & (CE | CF | X) & (CG | CH | Y) & (CI | CJ | Z) & (CK | CL | AA)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX) & (AY | AZ | BA) & (BB | BC | BD) & (BE | BF | BG) & "
        "(BH | BI | BJ) & (BK | BL | BM) & (BN | BO | BP) & (BQ | BR | BS) & "
        "(BT | BU | BV) & (BW | BX | BY) & (BZ | CA | CB) & (CC | CD | CE) & "
        "(CF | CG | CH) & (CI | CJ | CK) & (CL | A | Q)"
    )
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 95-vars")


def test_timed_shamir_matsat_mpspdz_100vars():
    # Add five more beyond 95: CM, CN, CO, CP, CQ = 100 total
    f1 = expr(
        "(A | B | C) & (D | E) & (F | G) & (H | I) & (J | K) & "
        "(L | M) & (N | O) & (P | Q) & (R | S) & (T | U) & "
        "(V | W) & (X | Y) & (Z | AA) & (AB | AC) & (AD | AE) & "
        "(AF | AG) & (AH | AI) & (AJ | AK) & (AL | AM) & (AN | AO) & "
        "(AP | AQ) & (AR | AS) & (AT | AU) & (AV | AW) & (AX | AY) & "
        "(AZ | BA) & (BB | BC) & (BD | BE) & (BF | BG) & (BH | A) & "
        "(BI | BJ) & (BK | BL) & (BM | BN) & (BO | BP) & (BQ | BR) & "
        "(BS | BT) & (BU | BV) & (BW | BX) & (BY | BZ) & (CA | CB) & "
        "(CC | CD) & (CE | CF) & (CG | CH) & (CI | CJ) & (CK | CL) & "
        "(CM | CN) & (CO | CP) & (CQ | A)"
    )
    f2 = expr(
        "(~A | C | H) & (I | ~D) & (E | F) & (G | J) & (K | M) & "
        "(N | ~H | O) & (P | Q | R) & (S | T | B) & (L | J) & (M | N) & "
        "(U | V) & (W | X | Y) & (Z | AA | B) & (AB | AC | D) & (AD | AE | E) & "
        "(AF | AG | H) & (AI | AJ) & (AK | AL | AM) & (AN | A | I) & "
        "(AO | AP | AQ) & (AR | AS | C) & (AT | AU | D) & (AV | AW | E) & "
        "(AX | AY | F) & (AZ | BA | G) & (BB | BC | H) & (BD | BE | I) & "
        "(BF | BG | J) & (BH | K | L) & (BI | BJ | M) & (BK | BL | N) & "
        "(BM | BN | O) & (BO | BP | P) & (BQ | BR | Q) & (BS | BT | R) & "
        "(BU | BV | S) & (BW | BX | T) & (BY | BZ | U) & (CA | CB | V) & "
        "(CC | CD | W) & (CE | CF | X) & (CG | CH | Y) & (CI | CJ | Z) & "
        "(CK | CL | AA) & (CM | CN | AB) & (CO | CP | AC) & (CQ | AD | AE)"
    )
    f3 = expr(
        "(~C | E | N) & (B | ~G | D) & (H | I | O) & (J | K | L) & (M | N | O) & "
        "(P | Q | R) & (S | T | A) & (B | C | D) & (E | F | G) & (H | I | J) & "
        "(U | V | W) & (X | Y | Z) & (AA | AB | AC) & (AD | AE | AF) & (AG | AH | AI) & "
        "(AJ | AK | AL) & (AM | AN | AO) & (AP | AQ | AR) & (AS | AT | AU) & "
        "(AV | AW | AX) & (AY | AZ | BA) & (BB | BC | BD) & (BE | BF | BG) & "
        "(BH | BI | BJ) & (BK | BL | BM) & (BN | BO | BP) & (BQ | BR | BS) & "
        "(BT | BU | BV) & (BW | BX | BY) & (BZ | CA | CB) & (CC | CD | CE) & "
        "(CF | CG | CH) & (CI | CJ | CK) & (CL | CM | CN) & (CO | CP | CQ)"
    )
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 100-vars")


# --- helpers for big-N tests ---


def _excel_var_names(n: int):
    """A, B, ..., Z, AA, AB, ... up to n names."""
    names = []
    i = 0
    while len(names) < n:
        x = i
        s = ""
        while True:
            s = chr(ord("A") + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        names.append(s)
        i += 1
    return names


# --- the big timed tests ---


def test_timed_shamir_matsat_mpspdz_110vars():
    f1, f2, f3 = _build_exprs(110)
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 110-vars")


def test_timed_shamir_matsat_mpspdz_120vars():
    f1, f2, f3 = _build_exprs(120)
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 120-vars")


def test_timed_shamir_matsat_mpspdz_130vars():
    f1, f2, f3 = _build_exprs(130)
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 130-vars")


def test_timed_shamir_matsat_mpspdz_140vars():
    f1, f2, f3 = _build_exprs(140)
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 140-vars")


def test_timed_shamir_matsat_mpspdz_150vars():
    f1, f2, f3 = _build_exprs(150)
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 150-vars")


def test_timed_shamir_matsat_mpspdz_160vars():
    f1, f2, f3 = _build_exprs(160)
    run_timed_mpspdz_test([f1, f2, f3], test_name="Shamir 160-vars")


def test_mat_sat_mpspdz_async_three_formula_sets():
    """Run three concurrent mat_sat_mpspdz_async calls using pre-reserved ports."""

    formula_sets = [
        [expr("A | B"), expr("~A | C"), expr("~B | C")],
        [expr("X | Y"), expr("~X | Y"), expr("~Y | Z")],
        [expr("P | Q"), expr("~P | R"), expr("~R | Q")],
    ]
    combined_formulas = [combine_formulas(fs) for fs in formula_sets]

    async def run_async_tests():
        reserved_ports = await reserve_ports_for_formula_sets(formula_sets)
        tasks = [
            mat_sat_mpspdz_async(formulas, port=port, timeout=120, protocol="shamir")
            for formulas, port in zip(formula_sets, reserved_ports)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_async_tests())

    for idx, (result, combined_formula) in enumerate(
        zip(results, combined_formulas), start=1
    ):
        assert not isinstance(
            result, Exception
        ), f"Async MP-SPDZ run {idx} raised {result}"
        assert result is not None, f"Async MP-SPDZ run {idx} returned None"
        assert verifier(result, combined_formula)


def test_mat_sat_mpspdz_async_three_formula_sets_sequential():
    """Run the same three formula sets sequentially without reserving ports in advance."""

    formula_sets = [
        [expr("A | B"), expr("~A | C"), expr("~B | C")],
        [expr("X | Y"), expr("~X | Y"), expr("~Y | Z")],
        [expr("P | Q"), expr("~P | R"), expr("~R | Q")],
    ]
    combined_formulas = [combine_formulas(fs) for fs in formula_sets]

    async def run_sequential_tests():
        results = []
        for formulas in formula_sets:
            result = await mat_sat_mpspdz_async(
                formulas, port=None, timeout=120, protocol="shamir"
            )
            results.append(result)
        return results

    results = asyncio.run(run_sequential_tests())

    for idx, (result, combined_formula) in enumerate(
        zip(results, combined_formulas), start=1
    ):
        assert result is not None, f"Sequential MP-SPDZ run {idx} returned None"
        assert verifier(result, combined_formula)


def test_mat_sat_mpspdz_async_unsatisfiable():
    """Verify that an unsatisfiable formula set returns None/False."""

    formula_set = [
        expr("A"),  # requires A to be True
        expr("B"),  # requires B to be True
        expr("~A | ~B"),  # forbids both A and B being True simultaneously
    ]

    async def run_unsat_test():
        return await mat_sat_mpspdz_async(formula_set, timeout=120, protocol="shamir")

    result = asyncio.run(run_unsat_test())
    assert result is None or result is False


def test_mat_sat_mpspdz_sync_unsatisfiable():
    """Verify that an unsatisfiable formula set returns None/False (sync version)."""

    formula_set = [
        expr("A"),  # requires A to be True
        expr("B"),  # requires B to be True
        expr("~A | ~B"),  # forbids both A and B being True simultaneously
    ]

    result = mat_sat_mpspdz(formula_set, protocol="shamir", debug=True)
    assert result is None or result is False

if __name__ == "__main__":
    pytest.main()
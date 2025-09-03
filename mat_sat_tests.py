from mat_sat import *
from utils4e import expr
from logic4e import to_cnf, conjuncts, prop_symbols


def test_simple_satisfiable():
    # (A) is trivially satisfiable → A must be True
    f = expr("A")
    result = mat_sat(f, seed=1)
    assert result != {}
    assert result["A"] is True


def test_simple_unsatisfiable():
    # (A & ~A) is always unsatisfiable → return {}
    f = expr("A & ~A")
    result = mat_sat(f, seed=1)
    assert result == False


def test_more_complex():
    # (A | B) & (C | ~D) & (~A | D)
    # one satisfying assignment is A=True, B=False, C=False, D=True
    f = expr("(A | B) & (C | ~D) & (~A | D)")
    result = mat_sat(f, seed=2)
    assert result != {}
    # Verify it really satisfies the clauses
    assert (result["A"] or result["B"]) is True
    assert (result["C"] or (not result["D"])) is True
    assert ((not result["A"]) or result["D"]) is True


def test_difficult_case():
    # Formula: (A | B) & (~A | C) & (~B | D) & (~C | ~D)
    # This forces a dependency chain:
    # - If A is False, then C must be True
    # - If B is False, then D must be True
    # - But (~C | ~D) forces at least one of C or D to be False
    #
    # One satisfying assignment is: A=True, B=False, C=False, D=True
    f = expr("(A | B) & (~A | C) & (~B | D) & (~C | ~D)")
    result = mat_sat(f, seed=3, max_try=10, max_itr=300)
    assert result != {}
    # Check that assignment satisfies each clause
    assert result["A"] or result["B"]
    assert (not result["A"]) or result["C"]
    assert (not result["B"]) or result["D"]
    assert (not result["C"]) or (not result["D"])


def test_non_cnf_expression():
    # Not in CNF: (A | (B & C)) & (~A | D)
    # CNF would distribute to: (A | B) & (A | C) & (~A | D)
    f = expr("(A | (B & C)) & (~A | D)")

    result = mat_sat(f, seed=5, max_try=10, max_itr=300)
    assert result != {}

    # Verify against the ORIGINAL (non-CNF) logic:
    # (A | (B & C)) must be True
    assert (result["A"] or (result["B"] and result["C"])) is True
    # (~A | D) must be True
    assert ((not result["A"]) or result["D"]) is True

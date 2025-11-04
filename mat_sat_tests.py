from mat_sat import *
from utils4e import expr
from utils import Expr
from logic4e import to_cnf, conjuncts, prop_symbols, pl_true, dpll_satisfiable
import asyncio


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


## tests for mat_sat_cpp
def test_mat_sat_cpp():
    f = expr("(A | (B & C)) & (~A | D)")
    result = mat_sat_cpp(f)
    cnf_f = to_cnf(f)
    assert verifier(result, f)

    assert verifier(result, f)


# def test_long_example():
#     f = expr(
#         "(X13 | X15 | ~X5) & (X5 | ~X13 | ~X9) & (~X2 | ~X13 | ~X9) & (~X16 | X18 | X19) & (~X6 | X14 | X5) & (~X7 | X4 | X11) & (~X15 | X19 | X14) & (X20 | ~X3 | ~X19) & (~X20 | ~X9 | ~X11) & (X2 | ~X6 | ~X10) & (X13 | ~X6 | X3) & (X9 | X11 | ~X8) & (~X9 | ~X19 | X7) & (~X17 | ~X20 | X12) & (~X17 | X4 | ~X16) & (X20 | ~X5 | ~X7) & (~X10 | ~X4 | X11) & (X5 | X9 | ~X1) & (X17 | ~X1 | X19) & (~X1 | ~X2 | ~X6) & (X15 | X17 | ~X19) & (X15 | ~X14 | X18) & (~X16 | ~X15 | X19) & (~X16 | X6 | ~X15) & (~X20 | X5 | ~X3) & (~X10 | X20 | X16) & (~X6 | X17 | ~X7) & (X7 | X2 | ~X16) & (~X18 | X5 | X13) & (~X17 | X13 | X12) & (~X14 | ~X6 | ~X12) & (X14 | ~X2 | ~X9) & (X3 | ~X14 | ~X17) & (~X1 | X18 | ~X6) & (X14 | ~X18 | ~X8) & (X7 | ~X3 | ~X19) & (~X18 | ~X20 | ~X5) & (X20 | X12 | X15) & (X5 | X3 | X15) & (X16 | ~X6 | ~X18) & (X8 | X5 | ~X18) & (X4 | X6 | ~X15) & (X6 | X3 | X4) & (X9 | ~X11 | ~X12) & (X12 | X9 | X5) & (X4 | X18 | ~X8) & (X16 | ~X8 | X1) & (X3 | X1 | ~X7) & (X15 | ~X9 | ~X4) & (~X5 | ~X3 | ~X10) & (~X16 | ~X12 | ~X19) & (X12 | ~X3 | ~X16) & (X4 | ~X18 | ~X6) & (X5 | ~X7 | ~X3) & (X15 | ~X1 | ~X5) & (~X16 | X9 | X10) & (~X9 | X17 | X5) & (~X2 | X4 | X10) & (X16 | X9 | ~X11) & (X1 | ~X7 | ~X15) & (~X20 | ~X8 | X3) & (X3 | X9 | X17) & (~X11 | X9 | X6) & (X8 | X16 | X19) & (X2 | X8 | ~X3) & (~X5 | X15 | X18) & (X1 | X16 | X2) & (~X18 | ~X11 | ~X9) & (X5 | X7 | ~X12) & (~X13 | ~X10 | X20) & (X11 | ~X20 | X1) & (~X13 | X19 | X2) & (X17 | ~X3 | X15) & (~X2 | X4 | X13) & (X5 | ~X19 | X12) & (~X12 | ~X5 | X7) & (X19 | ~X4 | X2) & (~X5 | ~X14 | X10) & (~X6 | ~X1 | ~X12) & (X20 | ~X18 | ~X11) & (X14 | X16 | X4) & (X5 | X12 | ~X10) & (X10 | X3 | ~X6) & (~X15 | ~X3 | X5) & (X12 | ~X13 | ~X1) & (X20 | ~X9 | ~X8) & (~X10 | X18 | ~X6) & (X16 | X12 | ~X18) & (~X14 | X15 | ~X2) & (X3 | X19 | X10) & (X15 | X20 | X13)"
#     )
#     result = dpll_satisfiable(f)
#     assert verifier(result, f)


# def test_mat_sat_cpp_long_example():
#     f = expr(
#         "(X13 | X15 | ~X5) & (X5 | ~X13) & (~X2 | ~X13 | ~X9) & (~X16 | X18 | X19) & (~X6 | X14 | X5) & (~X7 | X4 | X11) & (~X15 | X19 | X14) & (X20 | ~X3 | ~X19) & (~X20 | ~X9 | ~X11) & (X2 | ~X6 | ~X10) & (X13 | ~X6 | X3) & (X9 | X11 | ~X8) & (~X9 | ~X19 | X7) & (~X17 | ~X20 | X12) & (~X17 | X4 | ~X16) & (X20 | ~X5 | ~X7) & (~X10 | ~X4 | X11) & (X5 | X9 | ~X1) & (X17 | ~X1 | X19) & (~X1 | ~X2 | ~X6) & (X15 | X17 | ~X19) & (X15 | ~X14 | X18) & (~X16 | ~X15 | X19) & (~X16 | X6 | ~X15) & (~X20 | X5 | ~X3) & (~X10 | X20 | X16) & (~X6 | X17 | ~X7) & (X7 | X2 | ~X16) & (~X18 | X5 | X13) & (~X17 | X13 | X12) & (~X14 | ~X6 | ~X12) & (X14 | ~X2 | ~X9) & (X3 | ~X14 | ~X17) & (~X1 | X18 | ~X6) & (X14 | ~X18 | ~X8) & (X7 | ~X3 | ~X19) & (~X18 | ~X20 | ~X5) & (X20 | X12 | X15) & (X5 | X3 | X15) & (X16 | ~X6 | ~X18) & (X8 | X5 | ~X18) & (X4 | X6 | ~X15) & (X6 | X3 | X4) & (X9 | ~X11 | ~X12) & (X12 | X9 | X5) & (X4 | X18 | ~X8) & (X16 | ~X8 | X1) & (X3 | X1 | ~X7) & (X15 | ~X9 | ~X4) & (~X5 | ~X3 | ~X10) & (~X16 | ~X12 | ~X19) & (X12 | ~X3 | ~X16) & (X4 | ~X18 | ~X6) & (X5 | ~X7 | ~X3) & (X15 | ~X1 | ~X5) & (~X16 | X9 | X10) & (~X9 | X17 | X5) & (~X2 | X4 | X10) & (X16 | X9 | ~X11) & (X1 | ~X7 | ~X15) & (~X20 | ~X8 | X3) & (X3 | X9 | X17) & (~X11 | X9 | X6) & (X8 | X16 | X19) & (X2 | X8 | ~X3) & (~X5 | X15 | X18) & (X1 | X16 | X2) & (~X18 | ~X11 | ~X9) & (X5 | X7 | ~X12) & (~X13 | ~X10 | X20) & (X11 | ~X20 | X1) & (~X13 | X19 | X2) & (X17 | ~X3 | X15) & (~X2 | X4 | X13) & (X5 | ~X19 | X12) & (~X12 | ~X5 | X7) & (X19 | ~X4 | X2) & (~X5 | ~X14 | X10) & (~X6 | ~X1 | ~X12) & (X20 | ~X18 | ~X11) & (X14 | X16 | X4) & (X5 | X12 | ~X10) & (X10 | X3 | ~X6) & (~X15 | ~X3 | X5) & (X12 | ~X13 | ~X1) & (X20 | ~X9 | ~X8) & (~X10 | X18 | ~X6) & (X16 | X12 | ~X18) & (~X14 | X15 | ~X2) & (X3 | X19 | X10) & (X15 | X20 | X13)"
#     )
#     result = mat_sat_cpp(f)
#     assert verifier(result, f)
#     print(result)


# def test_mat_sat_cpp_unsat():
#     f = expr("A & ~A")
#     result = mat_sat_cpp(f)
#     assert result is None
#     assert not verifier(result, f)
#     print(result)


## tests for async mat_sat_mpspdz
def test_mat_sat_mpspdz_async_concurrent():
    """Test concurrent execution of async mat_sat_mpspdz with multiple independent calls."""

    async def run_concurrent_tests():
        # Create multiple independent formula sets that can run concurrently
        # Each should use its own temporary subfolder and port
        formula_sets = [
            [expr("A | B"), expr("C | D")],  # First concurrent call
            [expr("E | F"), expr("G | H")],  # Second concurrent call
            [expr("I | J"), expr("K | L")],  # Third concurrent call
            [expr("M | N"), expr("O | P")],  # Fourth concurrent call
        ]

        # Reserve ports upfront for all formula sets to avoid race conditions
        reserved_ports = await reserve_ports_for_formula_sets(formula_sets)
        assert len(reserved_ports) == len(formula_sets)

        # Run all calls concurrently with pre-reserved ports
        results = await asyncio.gather(
            *[
                mat_sat_mpspdz_async(formulas, port=port)
                for formulas, port in zip(formula_sets, reserved_ports)
            ]
        )

        # All should be satisfiable
        for i, result in enumerate(results):
            assert result is not None, f"Formula set {i} should be satisfiable"

        # Verify we got results from all concurrent calls
        assert len(results) == len(
            formula_sets
        ), "Should have results from all concurrent calls"

    # Run the async test
    asyncio.run(run_concurrent_tests())


def test_mat_sat_mpspdz_async_concurrent_unsat():
    """Test concurrent execution of async mat_sat_mpspdz with multiple unsatisfiable formulas."""

    async def run_concurrent_tests():
        formula_sets = [
            [expr("A & ~A"), expr("B & ~B")],
            [expr("C & ~C"), expr("D & ~D")],
        ]
        reserved_ports = await reserve_ports_for_formula_sets(formula_sets)
        results = await asyncio.gather(
            *[
                mat_sat_mpspdz_async(formulas, port=port)
                for formulas, port in zip(formula_sets, reserved_ports)
            ]
        )
        for result in results:
            assert result is None, "Unsatisfiable formulas should return None"

    asyncio.run(run_concurrent_tests())

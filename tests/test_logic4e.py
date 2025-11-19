import pytest
import sys
import os

# Add parent directory to path so we can import logic4e
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic4e import *
from utils4e import expr_handle_infix_ops, count

definite_clauses_KB = PropDefiniteKB()
for clause in [
    "(B & F)==>E",
    "(A & E & F)==>G",
    "(B & C)==>F",
    "(A & B)==>D",
    "(E & F)==>H",
    "(H & I)==>J",
    "A",
    "B",
    "C",
]:
    definite_clauses_KB.tell(expr(clause))


def test_is_symbol():
    assert is_symbol("x")
    assert is_symbol("X")
    assert is_symbol("N245")
    assert not is_symbol("")
    assert not is_symbol("1L")
    assert not is_symbol([1, 2, 3])


def test_is_var_symbol():
    assert is_var_symbol("xt")
    assert not is_var_symbol("Txt")
    assert not is_var_symbol("")
    assert not is_var_symbol("52")


def test_is_prop_symbol():
    assert not is_prop_symbol("xt")
    assert is_prop_symbol("Txt")
    assert not is_prop_symbol("")
    assert not is_prop_symbol("52")


def test_variables():
    assert variables(expr("F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)")) == {x, y, z}
    assert variables(expr("(x ==> y) & B(x, y) & A")) == {x, y}


def test_expr():
    assert repr(expr("P <=> Q(1)")) == "(P <=> Q(1))"
    assert repr(expr("P & Q | ~R(x, F(x))")) == "((P & Q) | ~R(x, F(x)))"
    assert expr_handle_infix_ops("P & Q ==> R & ~S") == "P & Q |'==>'| R & ~S"


def test_extend():
    assert extend({x: 1}, y, 2) == {x: 1, y: 2}


def test_subst():
    assert subst({x: 42, y: 0}, F(x) + y) == (F(42) + 0)


def test_PropKB():
    kb = PropKB()
    assert count(kb.ask(expr) for expr in [A, C, D, E, Q]) is 0
    kb.tell(A & E)
    assert kb.ask(A) == kb.ask(E) == {}
    kb.tell(E | "==>" | C)
    assert kb.ask(C) == {}
    kb.retract(E)
    assert kb.ask(E) is False
    assert kb.ask(C) is False


def test_wumpus_kb():
    # Statement: There is no pit in [1,1].
    assert wumpus_kb.ask(~P11) == {}

    # Statement: There is no pit in [1,2].
    assert wumpus_kb.ask(~P12) == {}

    # Statement: There is a pit in [2,2].
    assert wumpus_kb.ask(P22) is False

    # Statement: There is a pit in [3,1].
    assert wumpus_kb.ask(P31) is False

    # Statement: Neither [1,2] nor [2,1] contains a pit.
    assert wumpus_kb.ask(~P12 & ~P21) == {}

    # Statement: There is a pit in either [2,2] or [3,1].
    assert wumpus_kb.ask(P22 | P31) == {}


def test_basic_true_facts():
    kb = WumpusSATKB(2)
    assert kb.ask_if_true(~pit(1, 1)) is True
    assert kb.ask_if_true(~wumpus(1, 1)) is True
    assert kb.ask_if_true(location(1, 1, 0)) is True
    assert kb.ask_if_true(facing_east(0)) is True


def test_basic_non_entailed():
    kb = WumpusSATKB(2)
    assert kb.ask_if_true(pit(2, 2)) is False
    assert kb.ask_if_true(wumpus(2, 2)) is False
    assert kb.ask_if_true(facing_north(0)) is False


def test_is_definite_clause():
    assert is_definite_clause(expr("A & B & C & D ==> E"))
    assert is_definite_clause(expr("Farmer(Mac)"))
    assert not is_definite_clause(expr("~Farmer(Mac)"))
    assert is_definite_clause(expr("(Farmer(f) & Rabbit(r)) ==> Hates(f, r)"))
    assert not is_definite_clause(expr("(Farmer(f) & ~Rabbit(r)) ==> Hates(f, r)"))
    assert not is_definite_clause(expr("(Farmer(f) | Rabbit(r)) ==> Hates(f, r)"))


def test_parse_definite_clause():
    assert parse_definite_clause(expr("A & B & C & D ==> E")) == ([A, B, C, D], E)
    assert parse_definite_clause(expr("Farmer(Mac)")) == ([], expr("Farmer(Mac)"))
    assert parse_definite_clause(expr("(Farmer(f) & Rabbit(r)) ==> Hates(f, r)")) == (
        [expr("Farmer(f)"), expr("Rabbit(r)")],
        expr("Hates(f, r)"),
    )


def test_pl_true():
    assert pl_true(P, {}) is None
    assert pl_true(P, {P: False}) is False
    assert pl_true(P | Q, {P: True}) is True
    assert pl_true((A | B) & (C | D), {A: False, B: True, D: True}) is True
    assert pl_true((A & B) & (C | D), {A: False, B: True, D: True}) is False
    assert pl_true((A & B) | (A & C), {A: False, B: True, C: True}) is False
    assert pl_true((A | B) & (C | D), {A: True, D: False}) is None
    assert pl_true(P | P, {}) is None


def test_tt_true():
    assert tt_true(P | ~P)
    assert tt_true("~~P <=> P")
    assert not tt_true((P | ~Q) & (~P | Q))
    assert not tt_true(P & ~P)
    assert not tt_true(P & Q)
    assert tt_true((P | ~Q) | (~P | Q))
    assert tt_true("(A & B) ==> (A | B)")
    assert tt_true("((A & B) & C) <=> (A & (B & C))")
    assert tt_true("((A | B) | C) <=> (A | (B | C))")
    assert tt_true("(A ==> B) <=> (~B ==> ~A)")
    assert tt_true("(A ==> B) <=> (~A | B)")
    assert tt_true("(A <=> B) <=> ((A ==> B) & (B ==> A))")
    assert tt_true("~(A & B) <=> (~A | ~B)")
    assert tt_true("~(A | B) <=> (~A & ~B)")
    assert tt_true("(A & (B | C)) <=> ((A & B) | (A & C))")
    assert tt_true("(A | (B & C)) <=> ((A | B) & (A | C))")


def test_dpll():
    assert dpll_satisfiable(
        A
        & ~B
        & C
        & (A | ~D)
        & (~E | ~D)
        & (C | ~D)
        & (~A | ~F)
        & (E | ~F)
        & (~D | ~F)
        & (B | ~C | D)
        & (A | ~E | F)
        & (~A | E | D)
    ) == {B: False, C: True, A: True, F: False, D: True, E: False}
    assert dpll_satisfiable(A & B & ~C & D) == {C: False, A: True, D: True, B: True}
    assert dpll_satisfiable((A | (B & C)) | "<=>" | ((A | B) & (A | C))) == {
        C: True,
        A: True,
    } or {C: True, B: True}
    assert dpll_satisfiable(A | "<=>" | B) == {A: True, B: True}
    assert dpll_satisfiable(A & ~B) == {A: True, B: False}
    assert dpll_satisfiable(P & ~P) is False


def test_find_pure_symbol():
    assert find_pure_symbol([A, B, C], [A | ~B, ~B | ~C, C | A]) == (A, True)
    assert find_pure_symbol([A, B, C], [~A | ~B, ~B | ~C, C | A]) == (B, False)
    assert find_pure_symbol([A, B, C], [~A | B, ~B | ~C, C | A]) == (None, None)


def test_unit_clause_assign():
    assert unit_clause_assign(A | B | C, {A: True}) == (None, None)
    assert unit_clause_assign(B | C, {A: True}) == (None, None)
    assert unit_clause_assign(B | ~A, {A: True}) == (B, True)


def test_find_unit_clause():
    assert find_unit_clause([A | B | C, B | ~C, ~A | ~B], {A: True}) == (B, False)


def test_unify():
    assert unify(x, x, {}) == {}
    assert unify(x, 3, {}) == {x: 3}
    assert unify(x & 4 & y, 6 & y & 4, {}) == {x: 6, y: 4}
    assert unify(expr("A(x)"), expr("A(B)")) == {x: B}
    assert unify(expr("American(x) & Weapon(B)"), expr("American(A) & Weapon(y)")) == {
        x: A,
        y: B,
    }


def test_pl_fc_entails():
    assert pl_fc_entails(horn_clauses_KB, expr("Q"))
    assert pl_fc_entails(definite_clauses_KB, expr("G"))
    assert pl_fc_entails(definite_clauses_KB, expr("H"))
    assert not pl_fc_entails(definite_clauses_KB, expr("I"))
    assert not pl_fc_entails(definite_clauses_KB, expr("J"))
    assert not pl_fc_entails(horn_clauses_KB, expr("SomethingSilly"))


def test_tt_entails():
    assert tt_entails(P & Q, Q)
    assert not tt_entails(P | Q, Q)
    assert tt_entails(A & (B | C) & E & F & ~(P | Q), A & E & F & ~P & ~Q)
    assert not tt_entails(P | "<=>" | Q, Q)
    assert tt_entails((P | "==>" | Q) & P, Q)
    assert not tt_entails((P | "<=>" | Q) & ~P, Q)


def test_prop_symbols():
    assert prop_symbols(expr("x & y & z | A")) == {A}
    assert prop_symbols(expr("(x & B(z)) ==> Farmer(y) | A")) == {
        A,
        expr("Farmer(y)"),
        expr("B(z)"),
    }


def test_get_all_ordered_symbols():
    assert get_all_ordered_symbols(expr("X & Y & Z | A")) == [A, X, Y, Z]
    assert get_all_ordered_symbols(expr("(x & B(z)) ==> Farmer(y) | A")) == [
        A,
        expr("B(z)"),
        expr("Farmer(y)"),
    ]


def test_constant_symbols():
    assert constant_symbols(expr("x & y & z | A")) == {A}
    assert constant_symbols(expr("(x & B(z)) & Father(John) ==> Farmer(y) | A")) == {
        A,
        expr("John"),
    }


def test_predicate_symbols():
    assert predicate_symbols(expr("x & y & z | A")) == set()
    assert predicate_symbols(expr("(x & B(z)) & Father(John) ==> Farmer(y) | A")) == {
        ("B", 1),
        ("Father", 1),
        ("Farmer", 1),
    }
    assert predicate_symbols(
        expr("(x & B(x, y, z)) & F(G(x, y), x) ==> P(Q(R(x, y)), x, y, z)")
    ) == {("B", 3), ("F", 2), ("G", 2), ("P", 4), ("Q", 1), ("R", 2)}


def test_eliminate_implications():
    assert repr(eliminate_implications("A ==> (~B <== C)")) == "((~B | ~C) | ~A)"
    assert repr(eliminate_implications(A ^ B)) == "((A & ~B) | (~A & B))"
    assert repr(eliminate_implications(A & B | C & ~D)) == "((A & B) | (C & ~D))"


def test_dissociate():
    assert dissociate("&", [A & B]) == [A, B]
    assert dissociate("|", [A, B, C & D, P | Q]) == [A, B, C & D, P, Q]
    assert dissociate("&", [A, B, C & D, P | Q]) == [A, B, C, D, P | Q]


def test_associate():
    assert (
        repr(associate("&", [(A & B), (B | C), (B & C)])) == "(A & B & (B | C) & B & C)"
    )
    assert repr(associate("|", [A | (B | (C | (A & B)))])) == "(A | B | C | (A & B))"


def test_move_not_inwards():
    assert repr(move_not_inwards(~(A | B))) == "(~A & ~B)"
    assert repr(move_not_inwards(~(A & B))) == "(~A | ~B)"
    assert repr(move_not_inwards(~(~(A | ~B) | ~~C))) == "((A | ~B) & ~C)"


def test_distribute_and_over_or():
    def test_entailment(s, has_and=False):
        result = distribute_and_over_or(s)
        if has_and:
            assert result.op == "&"
        assert tt_entails(s, result)
        assert tt_entails(result, s)

    test_entailment((A & B) | C, True)
    test_entailment((A | B) & C, True)
    test_entailment((A | B) | C, False)
    test_entailment((A & B) | (C | D), True)


def test_to_cnf():
    assert (
        repr(to_cnf(wumpus_world_inference & ~expr("~P12")))
        == "((~P12 | B11) & (~P21 | B11) & (P12 | P21 | ~B11) & ~B11 & P12)"
    )
    assert (
        repr(to_cnf((P & Q) | (~P & ~Q)))
        == "((~P | P) & (~Q | P) & (~P | Q) & (~Q | Q))"
    )
    assert repr(to_cnf("A <=> B")) == "((A | ~B) & (B | ~A))"
    assert repr(to_cnf("B <=> (P1 | P2)")) == "((~P1 | B) & (~P2 | B) & (P1 | P2 | ~B))"
    assert repr(to_cnf("A <=> (B & C)")) == "((A | ~B | ~C) & (B | ~A) & (C | ~A))"
    assert repr(to_cnf("a | (b & c) | d")) == "((b | a | d) & (c | a | d))"
    assert repr(to_cnf("A & (B | (D & E))")) == "(A & (D | B) & (E | B))"
    assert (
        repr(to_cnf("A | (B | (C | (D & E)))")) == "((D | A | B | C) & (E | A | B | C))"
    )
    assert (
        repr(to_cnf("(A <=> ~B) ==> (C | ~D)"))
        == "((B | ~A | C | ~D) & (A | ~A | C | ~D) & (B | ~B | C | ~D) & (A | ~B | C | ~D))"
    )


def test_pl_resolution():
    assert pl_resolution(wumpus_kb, ~P11)
    assert pl_resolution(wumpus_kb, ~B11)
    assert not pl_resolution(wumpus_kb, P22)
    assert pl_resolution(horn_clauses_KB, A)
    assert pl_resolution(horn_clauses_KB, B)
    assert not pl_resolution(horn_clauses_KB, P)
    assert not pl_resolution(definite_clauses_KB, P)


def test_more_pl_resolution():
    # Helpers for fresh symbols
    X1, Y1, Z1 = expr("X1"), expr("Y1"), expr("Z1")
    R1, G1, R2, G2 = expr("R1"), expr("G1"), expr("R2"), expr("G2")

    # 1) XOR + implication: (A ⊕ B) ∧ B ∧ (B ⇒ C) ⊨ ~A and ⊨ C
    kb1 = PropKB()
    kb1.tell(A | B)  # at least one
    kb1.tell(~A | ~B)  # at most one   => XOR
    kb1.tell(B)
    kb1.tell(B | "==>" | C)
    assert pl_resolution(kb1, ~A)
    assert pl_resolution(kb1, C)
    assert not pl_resolution(kb1, A)

    # 2) Exactly-one-of {A,B,C} with routed consequences
    #    (A ⇒ X1), (B ⇒ Y1), (C ⇒ Z1) ⇒ ⊨ (X1 ∨ Y1 ∨ Z1), but not any single one
    kb2 = PropKB()
    kb2.tell(A | B | C)  # at least one
    kb2.tell(~A | ~B)
    kb2.tell(~A | ~C)
    kb2.tell(~B | ~C)  # pairwise at-most-one
    kb2.tell(A | "==>" | X1)
    kb2.tell(B | "==>" | Y1)
    kb2.tell(C | "==>" | Z1)
    assert pl_resolution(kb2, X1 | Y1 | Z1)
    assert not pl_resolution(kb2, X1)
    assert not pl_resolution(kb2, Y1)
    assert not pl_resolution(kb2, Z1)

    # 3) Simple 2-coloring on a 2-node graph with an edge:
    kb3 = PropKB()
    # node 1 exactly one of {R1,G1}
    kb3.tell(R1 | G1)
    kb3.tell(~R1 | ~G1)
    # node 2 exactly one of {R2,G2}
    kb3.tell(R2 | G2)
    kb3.tell(~R2 | ~G2)
    # edge constraint: different colors
    kb3.tell(~R1 | ~R2)
    kb3.tell(~G1 | ~G2)
    kb3.tell(R1)
    assert pl_resolution(kb3, G2)
    assert pl_resolution(kb3, ~R2)
    assert not pl_resolution(kb3, R2)

    # 4) “At-least-one triggers” to a single consequence:
    #    (A ∨ B ∨ C) ∧ (¬A ∨ D) ∧ (¬B ∨ D) ∧ (¬C ∨ D) ⊨ D
    kb4 = PropKB()
    kb4.tell(A | B | C)
    kb4.tell(~A | D)
    kb4.tell(~B | D)
    kb4.tell(~C | D)
    assert pl_resolution(kb4, D)

    # 5) Equivalence chain with negations:
    #    (A ⇔ ¬B) ∧ (B ⇔ ¬C) ∧ A ⊨ C and ⊨ ¬B (but not B)
    kb5 = PropKB()
    kb5.tell(A | "<=>" | ~B)
    kb5.tell(B | "<=>" | ~C)
    kb5.tell(A)
    assert pl_resolution(kb5, C)
    assert pl_resolution(kb5, ~B)
    assert not pl_resolution(kb5, B)

    # 6) Longer implication chain: A ⇒ B ⇒ C ⇒ D ⇒ E with A fact ⊨ E
    kb6 = PropKB()
    kb6.tell(A)
    kb6.tell(A | "==>" | B)
    kb6.tell(B | "==>" | C)
    kb6.tell(C | "==>" | D)
    kb6.tell(D | "==>" | E)
    assert pl_resolution(kb6, E)

    kb7 = PropKB()
    kb7.tell(A)
    kb7.tell(~A)
    assert pl_resolution(kb7, Z1)

    kb9 = PropKB()
    kb9.tell(A | B | C)
    kb9.tell(~A | D)
    kb9.tell(~B | E)
    assert not pl_resolution(kb9, D)


def test_standardize_variables():
    e = expr("F(a, b, c) & G(c, A, 23)")
    assert len(variables(standardize_variables(e))) == 3
    # assert variables(e).intersection(variables(standardize_variables(e))) == {}
    assert is_variable(standardize_variables(expr("x")))


def test_fol_bc_ask():
    def test_ask(query, kb=None):
        q = expr(query)
        test_variables = variables(q)
        answers = fol_bc_ask(kb or test_kb, q)
        return sorted(
            [
                dict((x, v) for x, v in list(a.items()) if x in test_variables)
                for a in answers
            ],
            key=repr,
        )

    assert repr(test_ask("Farmer(x)")) == "[{x: Mac}]"
    assert repr(test_ask("Human(x)")) == "[{x: Mac}, {x: MrsMac}]"
    assert repr(test_ask("Rabbit(x)")) == "[{x: MrsRabbit}, {x: Pete}]"
    assert repr(test_ask("Criminal(x)", crime_kb)) == "[{x: West}]"


def test_fol_fc_ask():
    def test_ask(query, kb=None):
        q = expr(query)
        test_variables = variables(q)
        answers = fol_fc_ask(kb or test_kb, q)
        return sorted(
            [
                dict((x, v) for x, v in list(a.items()) if x in test_variables)
                for a in answers
            ],
            key=repr,
        )

    assert repr(test_ask("Criminal(x)", crime_kb)) == "[{x: West}]"
    assert repr(test_ask("Enemy(x, America)", crime_kb)) == "[{x: Nono}]"
    assert repr(test_ask("Farmer(x)")) == "[{x: Mac}]"
    assert repr(test_ask("Human(x)")) == "[{x: Mac}, {x: MrsMac}]"
    assert repr(test_ask("Rabbit(x)")) == "[{x: MrsRabbit}, {x: Pete}]"


def test_d():
    assert d(x * x - x, x) == 2 * x - 1


def test_WalkSAT():
    def check_SAT(clauses, single_solution={}):
        # Make sure the solution is correct if it is returned by WalkSat
        # Sometimes WalkSat may run out of flips before finding a solution
        soln = WalkSAT(clauses)
        if soln:
            assert all(pl_true(x, soln) for x in clauses)
            if single_solution:  # Cross check the solution if only one exists
                assert all(pl_true(x, single_solution) for x in clauses)
                assert soln == single_solution

    # Test WalkSat for problems with solution
    check_SAT([A & B, A & C])
    check_SAT([A | B, P & Q, P & B])
    check_SAT([A & B, C | D, ~(D | P)], {A: True, B: True, C: True, D: False, P: False})
    check_SAT([A, B, ~C, D], {C: False, A: True, B: True, D: True})
    # Test WalkSat for problems without solution
    assert WalkSAT([A & ~A], 0.5, 100) is None
    assert WalkSAT([A & B, C | D, ~(D | B)], 0.5, 100) is None
    assert WalkSAT([A | B, ~A, ~(B | C), C | D, P | Q], 0.5, 100) is None
    assert WalkSAT([A | B, B & C, C | D, D & A, P, ~P], 0.5, 100) is None


def test_SAT_plan():
    transition = {
        "A": {"Left": "A", "Right": "B"},
        "B": {"Left": "A", "Right": "C"},
        "C": {"Left": "B", "Right": "C"},
    }
    assert SAT_plan("A", transition, "C", 2) is None
    assert SAT_plan("A", transition, "B", 3) == ["Right"]
    assert SAT_plan("C", transition, "A", 3) == ["Left", "Left"]

    transition = {
        (0, 0): {"Right": (0, 1), "Down": (1, 0)},
        (0, 1): {"Left": (1, 0), "Down": (1, 1)},
        (1, 0): {"Right": (1, 0), "Up": (1, 0), "Left": (1, 0), "Down": (1, 0)},
        (1, 1): {"Left": (1, 0), "Up": (0, 1)},
    }
    assert SAT_plan((0, 0), transition, (1, 1), 4) == ["Right", "Down"]


if __name__ == "__main__":
    pytest.main()

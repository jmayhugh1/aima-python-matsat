// test_matsat.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// include the solver implementation but avoid its demo main()
#define main matsat_demo_main
#include "matsat.cpp"
#undef main

// tiny helper to call mat_sat with params (seed / iters)
static std::optional<Assignment> run_sat(const std::string &cnf,
                                         uint64_t seed,
                                         int max_try = 8,
                                         int max_itr = 200)
{
    MatSatParams P;
    P.seed = seed;
    P.max_try = max_try;
    P.max_itr = max_itr;
    return mat_sat(cnf, P);
}

TEST_CASE("test_simple_satisfiable: (A) ⇒ A=true")
{
    auto res = run_sat("A", /*seed=*/1);
    REQUIRE(res.has_value());
    REQUIRE(res->find("A") != res->end());
    CHECK(res->at("A") == true);
}

TEST_CASE("test_simple_unsatisfiable: (A & ~A) ⇒ UNSAT")
{
    auto res = run_sat("A & ~A", /*seed=*/1);
    CHECK_FALSE(res.has_value());
}

TEST_CASE("test_more_complex: (A | B) & (C | ~D) & (~A | D)")
{
    auto res = run_sat("(A | B) & (C | ~D) & (~A | D)", /*seed=*/2);
    REQUIRE(res.has_value());

    const bool A = res->at("A");
    const bool B = res->at("B");
    const bool C = res->at("C");
    const bool D = res->at("D");

    CHECK((A || B));
    CHECK((C || (!D)));
    CHECK(((!A) || D));
}

TEST_CASE("test_difficult_case: (A | B) & (~A | C) & (~B | D) & (~C | ~D)")
{
    auto res = run_sat("(A | B) & (~A | C) & (~B | D) & (~C | ~D)",
                       /*seed=*/3, /*max_try=*/10, /*max_itr=*/300);
    REQUIRE(res.has_value());

    const bool A = res->at("A");
    const bool B = res->at("B");
    const bool C = res->at("C");
    const bool D = res->at("D");

    CHECK((A || B));
    CHECK(((!A) || C));
    CHECK(((!B) || D));
    CHECK(((!C) || (!D)));
}

TEST_CASE("test_non_cnf_expression (fed as CNF): (A | (B & C)) & (~A | D)")
{
    // Original (non-CNF): (A | (B & C)) & (~A | D)
    // CNF used here: (A | B) & (A | C) & (~A | D)
    auto res = run_sat("(A | B) & (A | C) & (~A | D)",
                       /*seed=*/5, /*max_try=*/10, /*max_itr=*/300);
    REQUIRE(res.has_value());

    const bool A = res->at("A");
    const bool B = res->at("B");
    const bool C = res->at("C");
    const bool D = res->at("D");

    // Verify against the ORIGINAL (non-CNF) logic
    CHECK((A || (B && C)));
    CHECK(((!A) || D));
}

TEST_CASE("80 variables, trivially SAT")
{
    std::string f =
        "(X1) & (X2) & (X3) & (X4) & (X5) & (X6) & (X7) & (X8) & (X9) & (X10) & "
        "(X11) & (X12) & (X13) & (X14) & (X15) & (X16) & (X17) & (X18) & (X19) & (X20) & "
        "(X21) & (X22) & (X23) & (X24) & (X25) & (X26) & (X27) & (X28) & (X29) & (X30) & "
        "(X31) & (X32) & (X33) & (X34) & (X35) & (X36) & (X37) & (X38) & (X39) & (X40) & "
        "(X41) & (X42) & (X43) & (X44) & (X45) & (X46) & (X47) & (X48) & (X49) & (X50) & "
        "(X51) & (X52) & (X53) & (X54) & (X55) & (X56) & (X57) & (X58) & (X59) & (X60) & "
        "(X61) & (X62) & (X63) & (X64) & (X65) & (X66) & (X67) & (X68) & (X69) & (X70) & "
        "(X71) & (X72) & (X73) & (X74) & (X75) & (X76) & (X77) & (X78) & (X79) & (X80)";
    auto res = run_sat(f, /*seed=*/123);
    REQUIRE(res.has_value());
    // Sample a few
    CHECK(res->at("X1") == true);
    CHECK(res->at("X40") == true);
    CHECK(res->at("X80") == true);
}

TEST_CASE("20 vars 91 clauses")
{
    std::string f = "(X13 | X15 | ~X5) & (X5 | ~X13 | ~X9) & (~X2 | ~X13 | ~X9) & (~X16 | X18 | X19) & (~X6 | X14 | X5) & (~X7 | X4 | X11) & (~X15 | X19 | X14) & (X20 | ~X3 | ~X19) & (~X20 | ~X9 | ~X11) & (X2 | ~X6 | ~X10) & (X13 | ~X6 | X3) & (X9 | X11 | ~X8) & (~X9 | ~X19 | X7) & (~X17 | ~X20 | X12) & (~X17 | X4 | ~X16) & (X20 | ~X5 | ~X7) & (~X10 | ~X4 | X11) & (X5 | X9 | ~X1) & (X17 | ~X1 | X19) & (~X1 | ~X2 | ~X6) & (X15 | X17 | ~X19) & (X15 | ~X14 | X18) & (~X16 | ~X15 | X19) & (~X16 | X6 | ~X15) & (~X20 | X5 | ~X3) & (~X10 | X20 | X16) & (~X6 | X17 | ~X7) & (X7 | X2 | ~X16) & (~X18 | X5 | X13) & (~X17 | X13 | X12) & (~X14 | ~X6 | ~X12) & (X14 | ~X2 | ~X9) & (X3 | ~X14 | ~X17) & (~X1 | X18 | ~X6) & (X14 | ~X18 | ~X8) & (X7 | ~X3 | ~X19) & (~X18 | ~X20 | ~X5) & (X20 | X12 | X15) & (X5 | X3 | X15) & (X16 | ~X6 | ~X18) & (X8 | X5 | ~X18) & (X4 | X6 | ~X15) & (X6 | X3 | X4) & (X9 | ~X11 | ~X12) & (X12 | X9 | X5) & (X4 | X18 | ~X8) & (X16 | ~X8 | X1) & (X3 | X1 | ~X7) & (X15 | ~X9 | ~X4) & (~X5 | ~X3 | ~X10) & (~X16 | ~X12 | ~X19) & (X12 | ~X3 | ~X16) & (X4 | ~X18 | ~X6) & (X5 | ~X7 | ~X3) & (X15 | ~X1 | ~X5) & (~X16 | X9 | X10) & (~X9 | X17 | X5) & (~X2 | X4 | X10) & (X16 | X9 | ~X11) & (X1 | ~X7 | ~X15) & (~X20 | ~X8 | X3) & (X3 | X9 | X17) & (~X11 | X9 | X6) & (X8 | X16 | X19) & (X2 | X8 | ~X3) & (~X5 | X15 | X18) & (X1 | X16 | X2) & (~X18 | ~X11 | ~X9) & (X5 | X7 | ~X12) & (~X13 | ~X10 | X20) & (X11 | ~X20 | X1) & (~X13 | X19 | X2) & (X17 | ~X3 | X15) & (~X2 | X4 | X13) & (X5 | ~X19 | X12) & (~X12 | ~X5 | X7) & (X19 | ~X4 | X2) & (~X5 | ~X14 | X10) & (~X6 | ~X1 | ~X12) & (X20 | ~X18 | ~X11) & (X14 | X16 | X4) & (X5 | X12 | ~X10) & (X10 | X3 | ~X6) & (~X15 | ~X3 | X5) & (X12 | ~X13 | ~X1) & (X20 | ~X9 | ~X8) & (~X10 | X18 | ~X6) & (X16 | X12 | ~X18) & (~X14 | X15 | ~X2) & (X3 | X19 | X10) & (X15 | X20 | X13)";
    auto res = run_sat(f, /*seed=*/42, /*max_try=*/20, /*max_itr=*/2000000);
    REQUIRE(res.has_value());
    REQUIRE(verifier(res.value(), parseCNF(f)));
    
}

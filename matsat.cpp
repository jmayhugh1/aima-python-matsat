// cppimport
/*<%
from cppimport.templating import setup_pybind11
setup_pybind11(cfg)
# macOS/Unix: C++17 + O3; add libc++ if your toolchain needs it
cfg["compiler_args"] += ["-std=c++17", "-O3"]
# If you hit libc++/linker issues on macOS, uncomment:
# cfg["compiler_args"] += ["-stdlib=libc++"]
# cfg["linker_args"]   += ["-stdlib=libc++"]
%>*/
// C++17 port of the Python MatSat solver (gradient-descent minimization).
// Expects CNF strings like "(A | ~B) & (C | D) & (~A | B)".
// Returns an assignment if satisfiable; otherwise std::nullopt.
//
// NOTE: This file implements a *minimal* CNF string parser. If you need full
// to-CNF support, plug in a proper logic parser/converter and feed its CNF
// output here.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ------------------------ small utils ------------------------

static inline void trimInPlace(std::string &s)
{
    auto notSpace = [](int ch)
    { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
}

static std::string stripped(const std::string &s)
{
    std::string t = s;
    trimInPlace(t);
    return t;
}

static std::string removeOuterParens(std::string s)
{
    trimInPlace(s);
    if (!s.empty() && s.front() == '(' && s.back() == ')')
    {
        // remove only one pair
        int depth = 0;
        bool ok = true;
        for (size_t i = 0; i < s.size(); ++i)
        {
            if (s[i] == '(')
                depth++;
            else if (s[i] == ')')
                depth--;
            if (i != s.size() - 1 && depth == 0)
            {
                ok = false;
                break;
            }
        }
        if (ok)
        {
            s = s.substr(1, s.size() - 2);
            trimInPlace(s);
        }
    }
    return s;
}

// split by a top-level delimiter (not inside parentheses)
static std::vector<std::string> splitTopLevel(const std::string &s, char delim)
{
    std::vector<std::string> parts;
    int depth = 0;
    size_t last = 0;
    for (size_t i = 0; i < s.size(); ++i)
    {
        char c = s[i];
        if (c == '(')
            depth++;
        else if (c == ')')
            depth--;
        else if (c == delim && depth == 0)
        {
            parts.emplace_back(s.substr(last, i - last));
            last = i + 1;
        }
    }
    parts.emplace_back(s.substr(last));
    for (auto &p : parts)
        trimInPlace(p);
    return parts;
}

// split by token " | " at top-level; we accept bare '|' too.
static std::vector<std::string> splitClauseOr(const std::string &s)
{
    // we just split by '|' ignoring parentheses depth
    std::vector<std::string> parts;
    int depth = 0;
    size_t last = 0;
    for (size_t i = 0; i < s.size(); ++i)
    {
        char c = s[i];
        if (c == '(')
            depth++;
        else if (c == ')')
            depth--;
        else if (c == '|' && depth == 0)
        {
            parts.emplace_back(s.substr(last, i - last));
            last = i + 1;
        }
    }
    parts.emplace_back(s.substr(last));
    for (auto &p : parts)
        trimInPlace(p);
    return parts;
}

// ------------------------ CNF types & parser ------------------------

struct Literal
{
    std::string var; // symbol name
    bool positive;   // true => x, false => ~x
};

using Clause = std::vector<Literal>;
using CNF = std::vector<Clause>;

// Parse a CNF string like "(A | ~B) & (C | D) & (~A | B)"
// Variables: [A-Za-z0-9_]+ (simple).
static CNF parseCNF(std::string formulaCNF)
{
    trimInPlace(formulaCNF);
    if (formulaCNF.empty())
        return {};

    // split into clauses by top-level '&'
    std::vector<std::string> clauses = splitTopLevel(formulaCNF, '&');

    CNF cnf;
    cnf.reserve(clauses.size());

    for (auto cl : clauses)
    {
        cl = removeOuterParens(cl);
        auto toks = splitClauseOr(cl);
        Clause clause;
        clause.reserve(toks.size());
        for (auto tok : toks)
        {
            tok = stripped(tok);
            // remove outer parens around a literal if present
            tok = removeOuterParens(tok);
            bool positive = true;
            if (!tok.empty() && tok[0] == '~')
            {
                positive = false;
                tok = tok.substr(1);
                trimInPlace(tok);
            }
            // variable name is remaining token
            if (tok.empty())
                continue;
            clause.push_back({tok, positive});
        }
        if (!clause.empty())
            cnf.push_back(std::move(clause));
    }
    return cnf;
}

// Collect unique symbol names and return sorted vector
static std::vector<std::string> symbolsOrder(const CNF &cnf)
{
    std::unordered_set<std::string> S;
    for (const auto &cl : cnf)
    {
        for (const auto &lit : cl)
            S.insert(lit.var);
    }
    std::vector<std::string> syms(S.begin(), S.end());
    std::sort(syms.begin(), syms.end());
    return syms;
}

// ------------------------ Linear algebra helpers ------------------------

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;

static Vec vecOnes(size_t n) { return Vec(n, 1.0); }

static Vec matVec(const Mat &M, const Vec &x)
{
    const size_t m = M.size();
    const size_t n = x.size();
    Vec out(m, 0.0);
    for (size_t i = 0; i < m; ++i)
    {
        const auto &row = M[i];
        double s = 0.0;
        // assume row.size() == n
        for (size_t j = 0; j < n; ++j)
            s += row[j] * x[j];
        out[i] = s;
    }
    return out;
}

static double dot(const Vec &a, const Vec &b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

static void clip01(Vec &u)
{
    for (auto &v : u)
    {
        if (v < 0.0)
            v = 0.0;
        else if (v > 1.0)
            v = 1.0;
    }
}

static Vec operator+(const Vec &a, const Vec &b)
{
    Vec c(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        c[i] = a[i] + b[i];
    return c;
}
static Vec operator-(const Vec &a, const Vec &b)
{
    Vec c(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        c[i] = a[i] - b[i];
    return c;
}
static Vec operator*(const Vec &a, const Vec &b)
{ // elementwise
    Vec c(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        c[i] = a[i] * b[i];
    return c;
}
static Vec operator*(double s, const Vec &a)
{
    Vec c(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        c[i] = s * a[i];
    return c;
}
static Vec operator*(const Vec &a, double s) { return s * a; }

static Mat matSub(const Mat &A, const Mat &B)
{
    Mat C = A;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[i].size(); ++j)
            C[i][j] -= B[i][j];
    return C;
}

// (Transpose(M) @ v), where M is m x n, v is length m, result is length n
static Vec matT_vec(const Mat &M, const Vec &v)
{
    if (M.empty())
        return {};
    size_t m = M.size(), n = M[0].size();
    Vec out(n, 0.0);
    for (size_t i = 0; i < m; ++i)
    {
        double vi = v[i];
        if (vi == 0.0)
            continue;
        const auto &row = M[i];
        for (size_t j = 0; j < n; ++j)
            out[j] += row[j] * vi;
    }
    return out;
}

// row sums (m x n) -> length m
static Vec rowSums(const Mat &M)
{
    Vec out(M.size(), 0.0);
    for (size_t i = 0; i < M.size(); ++i)
    {
        out[i] = std::accumulate(M[i].begin(), M[i].end(), 0.0);
    }
    return out;
}

// ------------------------ Build Q1, Q2 from CNF ------------------------

struct Qs
{
    Mat Q1;                        // m x n (positives)
    Mat Q2;                        // m x n (negatives)
    std::vector<std::string> syms; // ordered variables
};

static Qs build_Q1_Q2(const CNF &cnf)
{
    std::vector<std::string> syms = symbolsOrder(cnf);
    size_t n = syms.size();
    size_t m = cnf.size();
    Mat Q1(m, std::vector<double>(n, 0.0));
    Mat Q2(m, std::vector<double>(n, 0.0));

    // map symbol to column
    std::unordered_map<std::string, size_t> idx;
    for (size_t j = 0; j < n; ++j)
        idx[syms[j]] = j;

    for (size_t i = 0; i < m; ++i)
    {
        for (const auto &lit : cnf[i])
        {
            auto it = idx.find(lit.var);
            if (it == idx.end())
                continue;
            size_t j = it->second;
            if (lit.positive)
                Q1[i][j] = 1.0;
            else
                Q2[i][j] = 1.0;
        }
    }
    return {std::move(Q1), std::move(Q2), std::move(syms)};
}

// ------------------------ MatSat core ------------------------

struct MatSatParams
{
    double ell = 1.0;
    int max_try = 8;
    int max_itr = 200;
    double beta = 0.25;
    uint64_t seed = 0;
};

using Assignment = std::unordered_map<std::string, bool>;

static int unsat_count(const Mat &A, const Vec &Q2_ones, const Vec &u_bin)
{
    // c_b = Q2_ones + A @ u_bin ; count sum(1 - min(1,c_b))
    Vec c = Q2_ones + matVec(A, u_bin);
    int count = 0;
    for (double v : c)
    {
        double t = 1.0 - std::min(1.0, v);
        if (t > 1e-12)
            count += 1; // each unsatisfied clause contributes >= something
    }
    return count;
}

static std::optional<Assignment> mat_sat_cnf(const CNF &cnf, const MatSatParams &P = {})
{
    auto q = build_Q1_Q2(cnf);
    const auto &Q1 = q.Q1;
    const auto &Q2 = q.Q2;
    const auto &syms = q.syms;

    size_t n = syms.size();
    size_t m = cnf.size();
    if (n == 0)
        return Assignment{}; // empty assignment

    Mat A = matSub(Q1, Q2);    // (m, n)
    Vec Q2_ones = rowSums(Q2); // (m,)

    auto J = [&](const Vec &u) -> double
    {
        // J = sum(1 - min(1, Q2*1 + A u)) + (ell/2)||u*(1-u)||^2
        Vec c = Q2_ones + matVec(A, u);
        double clause_pen = 0.0;
        for (double v : c)
            clause_pen += (1.0 - std::min(1.0, v));
        Vec bin_term(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            bin_term[i] = u[i] * (1.0 - u[i]);
        double bin_norm2 = dot(bin_term, bin_term);
        return clause_pen + 0.5 * P.ell * bin_norm2;
    };

    auto grad = [&](const Vec &u) -> Vec
    {
        // ∇J = (Q2 - Q1)^T * mask + ell * (u*(1-u)*(1-2u))
        Vec c = Q2_ones + matVec(A, u); // (m,)
        Vec mask(m, 0.0);
        for (size_t i = 0; i < m; ++i)
            mask[i] = (c[i] < 1.0) ? 1.0 : 0.0;
        // (Q2 - Q1)^T @ mask = -(A^T @ mask)
        Vec term1 = matT_vec(A, mask);
        for (double &v : term1)
            v = -v;
        Vec d(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            d[i] = u[i] * (1.0 - u[i]) * (1.0 - 2.0 * u[i]);
        Vec out(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            out[i] = term1[i] + P.ell * d[i];
        return out;
    };

    std::mt19937_64 rng(P.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    Vec u(n, 0.0);
    for (auto &v : u)
        v = uni(rng); // random init in [0,1]

    Vec best_u_bin(n, 0.0);
    int best_err = std::numeric_limits<int>::max();

    for (int t = 0; t < P.max_try; ++t)
    {
        for (int k = 0; k < P.max_itr; ++k)
        {
            double Jval = J(u);
            Vec g = grad(u);
            double gnorm2 = dot(g, g);
            if (gnorm2 == 0.0)
                break;
            double alpha = Jval / gnorm2; // adaptive step
            for (size_t i = 0; i < n; ++i)
                u[i] -= alpha * g[i];
            clip01(u);

            Vec u_bin(n, 0.0);
            for (size_t i = 0; i < n; ++i)
                u_bin[i] = (u[i] >= 0.5) ? 1.0 : 0.0;
            int err = unsat_count(A, Q2_ones, u_bin);
            if (err < best_err)
            {
                best_err = err;
                best_u_bin = u_bin;
                if (err == 0)
                {
                    Assignment asg;
                    for (size_t j = 0; j < n; ++j)
                        asg[syms[j]] = (best_u_bin[j] > 0.5);
                    return asg;
                }
            }
        }
        // randomized restart / noise mixing
        for (size_t i = 0; i < n; ++i)
        {
            double noise = uni(rng);
            u[i] = (1.0 - P.beta) * u[i] + P.beta * noise;
        }
    }

    if (best_err == 0)
    {
        Assignment asg;
        for (size_t j = 0; j < n; ++j)
            asg[syms[j]] = (best_u_bin[j] > 0.5);
        return asg;
    }
    return std::nullopt; // unsatisfiable under this method
}

// Convenience wrapper: accept CNF string.
static std::optional<Assignment> mat_sat(const std::string &cnf_str, const MatSatParams &P = {})
{
    CNF cnf = parseCNF(cnf_str);
    return mat_sat_cnf(cnf, P);
}

// ------------------------ demo & parity with your prints ------------------------

static void printAssignment(const std::optional<Assignment> &opt)
{
    if (!opt.has_value())
    {
        std::cout << "False\n";
        return;
    }
    const auto &m = *opt;
    std::cout << "{";
    bool first = true;
    for (const auto &kv : m)
    {
        if (!first)
            std::cout << ", ";
        std::cout << kv.first << ": " << (kv.second ? "true" : "false");
        first = false;
    }
    std::cout << "}\n";
}

static bool verifier(Assignment asg, const CNF &cnf)
{
    for (size_t i = 0; i < cnf.size(); ++i)
    {
        const auto &cl = cnf[i];
        bool clause_satisfied = false;
        for (const auto &lit : cl)
        {
            auto it = asg.find(lit.var);
            if (it == asg.end())
                continue; // unassigned variable
            bool val = it->second;
            if ((lit.positive && val) || (!lit.positive && !val))
            {
                clause_satisfied = true;
                break;
            }
        }
        if (!clause_satisfied)
        {
            return false;
        }
    }
    return true;
}

// --- CLI main for matsat.cpp -----------------------------------------------
#include <fstream>
#include <sstream>

static void print_usage(const char *prog)
{
    std::cerr << "Usage:\n"
                 "  "
              << prog << " \"(A | ~B) & (C | D)\"\n"
                         "  "
              << prog << " -f formula.cnf.txt\n"
                         "  echo \"(~X | Y) & (X | ~Y)\" | "
              << prog << " --stdin\n\n"
                         "Options:\n"
                         "  -f, --file <path>     Read CNF from file (use '-' to read stdin)\n"
                         "      --stdin           Read CNF from stdin\n"
                         "      --ell=<float>     Binary penalty weight (default 1.0)\n"
                         "      --max-try=<int>   Random-restart tries (default 8)\n"
                         "      --max-itr=<int>   Iterations per try (default 200)\n"
                         "      --beta=<float>    Restart noise mix in [0,1] (default 0.25)\n"
                         "      --seed=<uint64>   RNG seed (default 0)\n"
                         "      --verify          Check the result actually satisfies the CNF\n"
                         "  -q, --quiet           Only print the assignment/False\n"
                         "  -h, --help            Show this help\n\n"
                         "Output:\n"
                         "  Prints an assignment map like {A: true, B: false} if SAT, or 'False' if UNSAT.\n";
}

static std::string read_all_stdin()
{
    std::ostringstream oss;
    oss << std::cin.rdbuf();
    return oss.str();
}

static std::string read_file(const std::string &path)
{
    if (path == "-")
        return read_all_stdin();
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

#ifndef MATSAT_CLI
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

static py::object mat_sat_py(const std::string& cnf,
                             double ell=1.0, int max_try=8, int max_itr=200,
                             double beta=0.25, uint64_t seed=0) {
    MatSatParams P; P.ell=ell; P.max_try=max_try; P.max_itr=max_itr; P.beta=beta; P.seed=seed;
    auto opt = mat_sat(cnf, P);
    if (!opt) return py::none();
    py::dict d;
    for (auto& kv : *opt) d[py::str(kv.first)] = kv.second;
    return d;
}

PYBIND11_MODULE(matsat, m) {
    m.doc() = "MatSat CNF solver (C++17)";
    m.def("mat_sat", &mat_sat_py,
          "cnf"_a, "ell"_a=1.0, "max_try"_a=8, "max_itr"_a=200,
          "beta"_a=0.25, "seed"_a=0);
}
#endif

#ifdef MATSAT_CLI
int main(int argc, char *argv[])
{
    std::string formula;   // CNF string
    std::string file_path; // optional file
    bool read_stdin_flag = false;
    bool verify_flag = false;
    bool quiet = false;

    MatSatParams P; // defaults already set in struct

    // --- parse args ---
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-q" || arg == "--quiet")
        {
            quiet = true;
        }
        else if (arg == "--verify")
        {
            verify_flag = true;
        }
        else if (arg == "--stdin")
        {
            read_stdin_flag = true;
        }
        else if (arg == "-f" || arg == "--file")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "error: missing path after " << arg << "\n";
                return 2;
            }
            file_path = argv[++i];
        }
        else if (arg.rfind("--ell=", 0) == 0)
        {
            P.ell = std::stod(arg.substr(6));
        }
        else if (arg.rfind("--max-try=", 0) == 0)
        {
            P.max_try = std::stoi(arg.substr(10));
        }
        else if (arg.rfind("--max-itr=", 0) == 0)
        {
            P.max_itr = std::stoi(arg.substr(10));
        }
        else if (arg.rfind("--beta=", 0) == 0)
        {
            P.beta = std::stod(arg.substr(7));
        }
        else if (arg.rfind("--seed=", 0) == 0)
        {
            P.seed = static_cast<uint64_t>(std::stoull(arg.substr(7)));
        }
        else if (arg.size() && arg[0] == '-')
        {
            std::cerr << "error: unknown option: " << arg << "\n";
            return 2;
        }
        else
        {
            // positional CNF string (only one allowed)
            if (!formula.empty())
            {
                std::cerr << "error: multiple positional arguments; did you mean to quote the formula?\n";
                return 2;
            }
            formula = arg;
        }
    }

    try
    {
        // acquire formula text
        if (!file_path.empty())
        {
            formula = read_file(file_path);
        }
        else if (read_stdin_flag && formula.empty())
        {
            formula = read_all_stdin();
        }

        // sanity check
        if (formula.empty())
        {
            print_usage(argv[0]);
            return 2;
        }

        // solve
        auto result = mat_sat(formula, P);

        if (!quiet)
        {
            std::cerr << "[MatSat] ell=" << P.ell
                      << " max_try=" << P.max_try
                      << " max_itr=" << P.max_itr
                      << " beta=" << P.beta
                      << " seed=" << P.seed << "\n";
        }

        if (!result.has_value())
        {
            std::cout << "False\n";
            return 1; // UNSAT / not found
        }

        // print assignment
        printAssignment(result);

        // optional verification against the exact parsed CNF
        if (verify_flag)
        {
            CNF cnf = parseCNF(formula);
            bool ok = verifier(*result, cnf);
            if (!ok)
            {
                if (!quiet)
                    std::cerr << "Verifier: assignment does NOT satisfy the CNF.\n";
                return 3;
            }
            else if (!quiet)
            {
                std::cerr << "Verifier: OK.\n";
            }
        }

        return 0; // SAT
    }
    catch (const std::exception &ex)
    {
        std::cerr << "error: " << ex.what() << "\n";
        return 2;
    }
}
#endif

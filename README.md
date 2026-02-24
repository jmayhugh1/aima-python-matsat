

# `aima-python` [![Build Status](https://travis-ci.org/aimacode/aima-python.svg?branch=master)](https://travis-ci.org/aimacode/aima-python) [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/aimacode/aima-python)


Python code for the book *[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu).* You can use this in conjunction with a course on AI, or for study on your own. We're looking for [solid contributors](https://github.com/aimacode/aima-python/blob/master/CONTRIBUTING.md) to help.

# Updates for 4th Edition

The 4th edition of the book as out now in 2020, and thus we are updating the code. All code here will reflect the 4th edition. Changes include:

- Move from Python 3.5 to 3.7.
- More emphasis on Jupyter (Ipython) notebooks.
- More projects using external packages (tensorflow, etc.).



# Structure of the Project

When complete, this project will have Python implementations for all the pseudocode algorithms in the book, as well as tests and examples of use. For each major topic, such as `search`, we provide the following  files:

- `search.ipynb` and `search.py`: Implementations of all the pseudocode algorithms, and necessary support functions/classes/data. The `.py` file is generated automatically from the `.ipynb` file; the idea is that it is easier to read the documentation in the `.ipynb` file.
- `search_XX.ipynb`: Notebooks that show how to use the code, broken out into various topics (the `XX`).
- `tests/test_search.py`: A lightweight test suite, using `assert` statements, designed for use with [`py.test`](http://pytest.org/latest/), but also usable on their own.

# Python 3.7 and up

The code for the 3rd edition was in Python 3.5; the current 4th edition code is in Python 3.7. It should also run in later versions, but does not run in Python 2. You can [install Python](https://www.python.org/downloads) or use a browser-based Python interpreter such as [repl.it](https://repl.it/languages/python3).
You can run the code in an IDE, or from the command line with `python -i filename.py` where the `-i` option puts you in an interactive loop where you can run Python functions. All notebooks are available in a [binder environment](http://mybinder.org/repo/aimacode/aima-python). Alternatively, visit [jupyter.org](http://jupyter.org/) for instructions on setting up your own Jupyter notebook environment.

Features from Python 3.6 and 3.7 that we will be using for this version of the code:
- [f-strings](https://docs.python.org/3.6/whatsnew/3.6.html#whatsnew36-pep498): all string formatting should be done with `f'var = {var}'`, not with `'var = {}'.format(var)` nor `'var = %s' % var`.
- [`typing` module](https://docs.python.org/3.7/library/typing.html): declare functions with type hints: `def successors(state) -> List[State]:`; that is, give type declarations, but omit them when it is obvious. I don't need to say `state: State`, but in another context it would make sense to say `s: State`.
- Underscores in numerics: write a million as `1_000_000` not as `1000000`.
- [`dataclasses` module](https://docs.python.org/3.7/library/dataclasses.html#module-dataclasses): replace `namedtuple` with `dataclass`.


[//]: # (There is a sibling [aima-docker]https://github.com/rajatjain1997/aima-docker project that shows you how to use docker containers to run more complex problems in more complex software environments.)


## Installation Guide

To download the repository:

`git clone https://github.com/aimacode/aima-python.git`

Then you need to install the basic dependencies to run the project on your system:

```
cd aima-python
pip install -r requirements.txt
```

You also need to fetch the datasets from the [`aima-data`](https://github.com/aimacode/aima-data) repository:

```
git submodule init
git submodule update
```

Wait for the datasets to download, it may take a while. Once they are downloaded, you need to install `pytest`, so that you can run the test suite:

`pip install pytest`

Then to run the tests:

`py.test`

And you are good to go!


# Index of Algorithms

Here is a table of algorithms, the figure, name of the algorithm in the book and in the repository, and the file where they are implemented in the repository. This chart was made for the third edition of the book and is being updated for the upcoming fourth edition. Empty implementations are a good place for contributors to look for an issue. The [aima-pseudocode](https://github.com/aimacode/aima-pseudocode) project describes all the algorithms from the book. An asterisk next to the file name denotes the algorithm is not fully implemented. Another great place for contributors to start is by adding tests and writing on the notebooks. You can see which algorithms have tests and notebook sections below. If the algorithm you want to work on is covered, don't worry! You can still add more tests and provide some examples of use in the notebook!

| **Figure** | **Name (in 3<sup>rd</sup> edition)** | **Name (in repository)** | **File** | **Tests** | **Notebook**
|:-------|:----------------------------------|:------------------------------|:--------------------------------|:-----|:---------|
| 2      | Random-Vacuum-Agent               | `RandomVacuumAgent`           | [`agents.py`][agents]           | Done | Included |
| 2      | Model-Based-Vacuum-Agent          | `ModelBasedVacuumAgent`       | [`agents.py`][agents]           | Done | Included |
| 2.1    | Environment                       | `Environment`                 | [`agents.py`][agents]           | Done | Included |
| 2.1    | Agent                             | `Agent`                       | [`agents.py`][agents]           | Done | Included |
| 2.3    | Table-Driven-Vacuum-Agent         | `TableDrivenVacuumAgent`      | [`agents.py`][agents]           | Done | Included |
| 2.7    | Table-Driven-Agent                | `TableDrivenAgent`            | [`agents.py`][agents]           | Done | Included |
| 2.8    | Reflex-Vacuum-Agent               | `ReflexVacuumAgent`           | [`agents.py`][agents]           | Done | Included |
| 2.10   | Simple-Reflex-Agent               | `SimpleReflexAgent`           | [`agents.py`][agents]           | Done | Included |
| 2.12   | Model-Based-Reflex-Agent          | `ReflexAgentWithState`        | [`agents.py`][agents]           | Done | Included |
| 3      | Problem                           | `Problem`                     | [`search.py`][search]           | Done | Included |
| 3      | Node                              | `Node`                        | [`search.py`][search]           | Done | Included |
| 3      | Queue                             | `Queue`                       | [`utils.py`][utils]             | Done | No Need  |
| 3.1    | Simple-Problem-Solving-Agent      | `SimpleProblemSolvingAgent`   | [`search.py`][search]           | Done | Included |
| 3.2    | Romania                           | `romania`                     | [`search.py`][search]           | Done | Included |
| 3.7    | Tree-Search                       | `depth/breadth_first_tree_search`                 | [`search.py`][search]           | Done | Included |
| 3.7    | Graph-Search                      | `depth/breadth_first_graph_search`                | [`search.py`][search]           | Done | Included |
| 3.11   | Breadth-First-Search              | `breadth_first_graph_search`  | [`search.py`][search]           | Done | Included |
| 3.14   | Uniform-Cost-Search               | `uniform_cost_search`         | [`search.py`][search]           | Done | Included |
| 3.17   | Depth-Limited-Search              | `depth_limited_search`        | [`search.py`][search]           | Done | Included |
| 3.18   | Iterative-Deepening-Search        | `iterative_deepening_search`  | [`search.py`][search]           | Done | Included |
| 3.22   | Best-First-Search                 | `best_first_graph_search`     | [`search.py`][search]           | Done | Included |
| 3.24   | A\*-Search                        | `astar_search`                | [`search.py`][search]           | Done | Included |
| 3.26   | Recursive-Best-First-Search       | `recursive_best_first_search` | [`search.py`][search]           | Done | Included |
| 4.2    | Hill-Climbing                     | `hill_climbing`               | [`search.py`][search]           | Done | Included |
| 4.5    | Simulated-Annealing               | `simulated_annealing`         | [`search.py`][search]           | Done | Included |
| 4.8    | Genetic-Algorithm                 | `genetic_algorithm`           | [`search.py`][search]           | Done | Included |
| 4.11   | And-Or-Graph-Search               | `and_or_graph_search`         | [`search.py`][search]           | Done | Included |
| 4.21   | Online-DFS-Agent                  | `online_dfs_agent`            | [`search.py`][search]           | Done | Included |
| 4.24   | LRTA\*-Agent                      | `LRTAStarAgent`               | [`search.py`][search]           | Done | Included |
| 5.3    | Minimax-Decision                  | `minimax_decision`            | [`games.py`][games]             | Done | Included |
| 5.7    | Alpha-Beta-Search                 | `alphabeta_search`            | [`games.py`][games]             | Done | Included |
| 6      | CSP                               | `CSP`                         | [`csp.py`][csp]                 | Done | Included |
| 6.3    | AC-3                              | `AC3`                         | [`csp.py`][csp]                 | Done | Included |
| 6.5    | Backtracking-Search               | `backtracking_search`         | [`csp.py`][csp]                 | Done | Included |
| 6.8    | Min-Conflicts                     | `min_conflicts`               | [`csp.py`][csp]                 | Done | Included |
| 6.11   | Tree-CSP-Solver                   | `tree_csp_solver`             | [`csp.py`][csp]                 | Done | Included |
| 7      | KB                                | `KB`                          | [`logic.py`][logic]             | Done | Included |
| 7.1    | KB-Agent                          | `KB_AgentProgram`             | [`logic.py`][logic]             | Done | Included |
| 7.7    | Propositional Logic Sentence      | `Expr`                        | [`utils.py`][utils]             | Done | Included |
| 7.10   | TT-Entails                        | `tt_entails`                  | [`logic.py`][logic]             | Done | Included |
| 7.12   | PL-Resolution                     | `pl_resolution`               | [`logic.py`][logic]             | Done | Included |
| 7.14   | Convert to CNF                    | `to_cnf`                      | [`logic.py`][logic]             | Done | Included |
| 7.15   | PL-FC-Entails?                    | `pl_fc_entails`               | [`logic.py`][logic]             | Done | Included |
| 7.17   | DPLL-Satisfiable?                 | `dpll_satisfiable`            | [`logic.py`][logic]             | Done | Included |
| 7.18   | WalkSAT                           | `WalkSAT`                     | [`logic.py`][logic]             | Done | Included |
| 7.20   | Hybrid-Wumpus-Agent               | `HybridWumpusAgent`           |                                 |      |          |
| 7.22   | SATPlan                           | `SAT_plan`                    | [`logic.py`][logic]             | Done | Included |
| 9      | Subst                             | `subst`                       | [`logic.py`][logic]             | Done | Included |
| 9.1    | Unify                             | `unify`                       | [`logic.py`][logic]             | Done | Included |
| 9.3    | FOL-FC-Ask                        | `fol_fc_ask`                  | [`logic.py`][logic]             | Done | Included |
| 9.6    | FOL-BC-Ask                        | `fol_bc_ask`                  | [`logic.py`][logic]             | Done | Included |
| 10.1   | Air-Cargo-problem                 | `air_cargo`                   | [`planning.py`][planning]       | Done | Included |
| 10.2   | Spare-Tire-Problem                | `spare_tire`                  | [`planning.py`][planning]       | Done | Included |
| 10.3   | Three-Block-Tower                 | `three_block_tower`           | [`planning.py`][planning]       | Done | Included |
| 10.7   | Cake-Problem                      | `have_cake_and_eat_cake_too`  | [`planning.py`][planning]       | Done | Included |
| 10.9   | Graphplan                         | `GraphPlan`                   | [`planning.py`][planning]       | Done | Included |
| 10.13  | Partial-Order-Planner             | `PartialOrderPlanner`         | [`planning.py`][planning]       | Done | Included |
| 11.1   | Job-Shop-Problem-With-Resources   | `job_shop_problem`            | [`planning.py`][planning]       | Done | Included |
| 11.5   | Hierarchical-Search               | `hierarchical_search`         | [`planning.py`][planning]       | Done | Included |
| 11.8   | Angelic-Search                    | `angelic_search`              | [`planning.py`][planning]       | Done | Included |
| 11.10  | Doubles-tennis                    | `double_tennis_problem`       | [`planning.py`][planning]       | Done | Included |
| 13     | Discrete Probability Distribution | `ProbDist`                    | [`probability.py`][probability] | Done | Included |
| 13.1   | DT-Agent                          | `DTAgent`                     | [`probability.py`][probability] | Done | Included |
| 14.9   | Enumeration-Ask                   | `enumeration_ask`             | [`probability.py`][probability] | Done | Included |
| 14.11  | Elimination-Ask                   | `elimination_ask`             | [`probability.py`][probability] | Done | Included |
| 14.13  | Prior-Sample                      | `prior_sample`                | [`probability.py`][probability] | Done | Included |
| 14.14  | Rejection-Sampling                | `rejection_sampling`          | [`probability.py`][probability] | Done | Included |
| 14.15  | Likelihood-Weighting              | `likelihood_weighting`        | [`probability.py`][probability] | Done | Included |
| 14.16  | Gibbs-Ask                         | `gibbs_ask`                   | [`probability.py`][probability] | Done | Included |
| 15.4   | Forward-Backward                  | `forward_backward`            | [`probability.py`][probability] | Done | Included |
| 15.6   | Fixed-Lag-Smoothing               | `fixed_lag_smoothing`         | [`probability.py`][probability] | Done | Included |
| 15.17  | Particle-Filtering                | `particle_filtering`          | [`probability.py`][probability] | Done | Included |
| 16.9   | Information-Gathering-Agent       | `InformationGatheringAgent`   | [`probability.py`][probability] | Done | Included |
| 17.4   | Value-Iteration                   | `value_iteration`             | [`mdp.py`][mdp]                 | Done | Included |
| 17.7   | Policy-Iteration                  | `policy_iteration`            | [`mdp.py`][mdp]                 | Done | Included |
| 17.9   | POMDP-Value-Iteration             | `pomdp_value_iteration`       | [`mdp.py`][mdp]                 | Done | Included |
| 18.5   | Decision-Tree-Learning            | `DecisionTreeLearner`         | [`learning.py`][learning]       | Done | Included |
| 18.8   | Cross-Validation                  | `cross_validation`            | [`learning.py`][learning]\*     |      |          |
| 18.11  | Decision-List-Learning            | `DecisionListLearner`         | [`learning.py`][learning]\*     |      |          |
| 18.24  | Back-Prop-Learning                | `BackPropagationLearner`      | [`learning.py`][learning]       | Done | Included |
| 18.34  | AdaBoost                          | `AdaBoost`                    | [`learning.py`][learning]       | Done | Included |
| 19.2   | Current-Best-Learning             | `current_best_learning`       | [`knowledge.py`](knowledge.py)  | Done | Included |
| 19.3   | Version-Space-Learning            | `version_space_learning`      | [`knowledge.py`](knowledge.py)  | Done | Included |
| 19.8   | Minimal-Consistent-Det            | `minimal_consistent_det`      | [`knowledge.py`](knowledge.py)  | Done | Included |
| 19.12  | FOIL                              | `FOIL_container`              | [`knowledge.py`](knowledge.py)  | Done | Included |
| 21.2   | Passive-ADP-Agent                 | `PassiveADPAgent`             | [`rl.py`][rl]                   | Done | Included |
| 21.4   | Passive-TD-Agent                  | `PassiveTDAgent`              | [`rl.py`][rl]                   | Done | Included |
| 21.8   | Q-Learning-Agent                  | `QLearningAgent`              | [`rl.py`][rl]                   | Done | Included |
| 22.1   | HITS                              | `HITS`                        | [`nlp.py`][nlp]                 | Done | Included |
| 23     | Chart-Parse                       | `Chart`                       | [`nlp.py`][nlp]                 | Done | Included |
| 23.5   | CYK-Parse                         | `CYK_parse`                   | [`nlp.py`][nlp]                 | Done | Included |
| 25.9   | Monte-Carlo-Localization          | `monte_carlo_localization`    | [`probability.py`][probability] | Done | Included |


# Index of data structures

Here is a table of the implemented data structures, the figure, name of the implementation in the repository, and the file where they are implemented.

| **Figure** | **Name (in repository)** | **File** |
|:-------|:--------------------------------|:--------------------------|
| 3.2    | romania_map                     | [`search.py`][search]     |
| 4.9    | vacumm_world                    | [`search.py`][search]     |
| 4.23   | one_dim_state_space             | [`search.py`][search]     |
| 6.1    | australia_map                   | [`search.py`][search]     |
| 7.13   | wumpus_world_inference          | [`logic.py`][logic]       |
| 7.16   | horn_clauses_KB                 | [`logic.py`][logic]       |
| 17.1   | sequential_decision_environment | [`mdp.py`][mdp]           |
| 18.2   | waiting_decision_tree           | [`learning.py`][learning] |


## Private Path Query: Concise Position-Bit Encoding

The private path query logic now uses a logarithmic state encoding for position.
Instead of one Boolean per vertex (`At(t, v)`), we represent the vertex id at time `t`
in binary using `PosBit(t, k)`, where `k=0` is the least-significant bit.

### Intuition for newcomers

The old one-hot representation stores a position by turning on exactly one bit from a
length-`V` vector. The new representation stores the same position as a binary number
using only `ceil(log2(V))` bits.

At each timestep:

- one-hot: `[0, 0, 1, 0, 0, 0, 0, 0]` means vertex `2`
- bit encoding (`V=8`, 3 bits): `[0, 1, 0]` means vertex `2`

Both represent one vertex; the bit form just needs fewer variables.

### Worked comparison: `T = 8`, `V = 8`

Assume dense directed moves (`u != v`), so there are `8 * 7 = 56` directed pairs.

#### Old encoding (one-hot + explicit action vars)

Variables:

- `At(t, v)`: `(T+1)*V = 9*8 = 72`
- `Move(t, u, v)`: `T*V*(V-1) = 8*56 = 448`
- `Wait(t, v)`: `T*V = 8*8 = 64`
- `Active(u, v)`: `56`
- `Allowed(u, v)`: `56`
- total: `72 + 448 + 64 + 56 + 56 = 696`

Large clause source (before CNF expansion of other formulas):

- exactly-one on `At` at each time: 9 AMO/ALO groups over 8 literals each  
  (`1 + C(8,2) = 29` clauses per time) -> `9*29 = 261`
- exactly-one action at each `t` over `56 + 8 = 64` literals  
  (`1 + C(64,2) = 2017` clauses per time) -> `8*2017 = 16,136`

So the action AMO dominates quickly.

#### New encoding (position bits + direct transitions)

Here `b = ceil(log2(8)) = 3`.

Variables:

- `PosBit(t, k)`: `(T+1)*b = 9*3 = 27`
- `Active(u, v)`: `56`
- `Allowed(u, v)`: `56`
- total: `27 + 56 + 56 = 139`

Compared to 696 previously, that is about a **5x reduction in variable count** for this
example.

Also, because `V=8` is a power of two, there are **no invalid code range clauses**.

### Mathematical formulation of the bit encoding

#### Definitions

Let \(V\) be the number of vertices (labeled \(0, 1, \ldots, V-1\)) and let
\(b = \lceil \log_2 V \rceil\) be the number of bits needed to represent any vertex.

Define the Boolean variable:

\[
\text{PosBit}(t, k) \in \{\text{True}, \text{False}\}
\quad \text{for } t \in \{0, \ldots, T\},\; k \in \{0, \ldots, b-1\}
\]

The **position at time \(t\)** is reconstructed as:

\[
\text{pos}(t) = \sum_{k=0}^{b-1} 2^k \cdot [\![\text{PosBit}(t, k)]\!]
\]

where \([\![P]\!] = 1\) if \(P\) is true, else \(0\).

#### Equality predicate

To express "\(\text{pos}(t) = u\)" for a constant vertex \(u\), let \(u_k\) denote
the \(k\)-th bit of \(u\) (i.e., \(u_k = \lfloor u / 2^k \rfloor \mod 2\)). Then:

\[
\text{pos}(t) = u
\quad\Longleftrightarrow\quad
\bigwedge_{k=0}^{b-1} \bigl(\text{PosBit}(t, k) = u_k\bigr)
\]

As a propositional formula this is a conjunction of literals:

\[
\bigwedge_{k=0}^{b-1}
\begin{cases}
\text{PosBit}(t, k) & \text{if } u_k = 1 \\
\lnot\,\text{PosBit}(t, k) & \text{if } u_k = 0
\end{cases}
\]

#### Range restriction

When \(V\) is not a power of two, codes \(c \in \{V, V+1, \ldots, 2^b - 1\}\) are
invalid. For each such \(c\), add a clause forbidding that exact bit pattern:

\[
\bigvee_{k=0}^{b-1}
\begin{cases}
\lnot\,\text{PosBit}(t, k) & \text{if } c_k = 1 \\
\text{PosBit}(t, k) & \text{if } c_k = 0
\end{cases}
\]

This clause is the negation of "\(\text{pos}(t) = c\)", forcing at least one bit to
differ from \(c\).

#### Transition feasibility

Let \(\text{adj}(u)\) be the set of successors of vertex \(u\), including \(u\)
itself (to allow waiting). The constraint is:

\[
\text{pos}(t) = u
\;\;\Longrightarrow\;\;
\bigvee_{v \in \text{adj}(u)} \bigl(\text{pos}(t+1) = v\bigr)
\]

Expanding definitions and converting to CNF produces the actual clauses.

#### Start and goal fixing

To force \(\text{pos}(0) = s\) and \(\text{pos}(T) = g\), emit **unit clauses** for
each bit:

\[
\text{PosBit}(0, k) \text{ or } \lnot\text{PosBit}(0, k)
\quad\text{depending on } s_k
\]

and similarly for \(g\) at time \(T\).

---

### Worked example: \(V = 5\), \(T = 2\)

Since \(V = 5\), we have \(b = \lceil \log_2 5 \rceil = 3\) bits.

The vertices and their binary codes are:

| vertex | binary (\(k = 2, 1, 0\)) |
|--------|--------------------------|
| 0      | 000                      |
| 1      | 001                      |
| 2      | 010                      |
| 3      | 011                      |
| 4      | 100                      |

Invalid codes (must be forbidden): 5 = 101, 6 = 110, 7 = 111.

#### Encoding a position

Suppose the agent is at vertex 3 at time \(t = 1\). Then:

\[
\text{pos}(1) = 3 = 0 \cdot 4 + 1 \cdot 2 + 1 \cdot 1
\]

The corresponding literal values are:

- \(\text{PosBit}(1, 0) = \text{True}\) (bit 0 of 3 is 1)
- \(\text{PosBit}(1, 1) = \text{True}\) (bit 1 of 3 is 1)
- \(\text{PosBit}(1, 2) = \text{False}\) (bit 2 of 3 is 0)

The equality "\(\text{pos}(1) = 3\)" is the conjunction:

\[
\text{PosBit}(1, 0) \;\land\; \text{PosBit}(1, 1) \;\land\; \lnot\text{PosBit}(1, 2)
\]

#### Why AMO (at-most-one) is implicit

In the old one-hot encoding, we needed explicit constraints to ensure the agent is at
exactly one vertex per timestep:

- **ALO (at-least-one):** \(\text{At}(t, 0) \lor \text{At}(t, 1) \lor \cdots \lor \text{At}(t, V-1)\)
- **AMO (at-most-one):** \(\lnot\text{At}(t, i) \lor \lnot\text{At}(t, j)\) for all \(i < j\)

The AMO constraints alone require \(\binom{V}{2} = O(V^2)\) clauses per timestep.

**With bit encoding, AMO is free.** Here's why:

Any Boolean assignment to the \(b\) bits \(\text{PosBit}(t, 0), \ldots, \text{PosBit}(t, b-1)\)
defines exactly one integer via:

\[
\text{pos}(t) = \sum_{k=0}^{b-1} 2^k \cdot [\![\text{PosBit}(t, k)]\!]
\]

This is a **bijection** between \(\{0,1\}^b\) and \(\{0, 1, \ldots, 2^b - 1\}\).
There is no way for the bits to simultaneously represent two different vertices.

For example, with \(b = 3\) bits at time \(t = 1\):

| Assignment                    | Decoded vertex |
|-------------------------------|----------------|
| (False, False, False)         | 0              |
| (True, False, False)          | 1              |
| (False, True, False)          | 2              |
| (True, True, False)           | 3              |
| ...                           | ...            |

Each row is one unique assignment → one unique vertex. The agent cannot be at vertex 2
and vertex 3 simultaneously because that would require \(\text{PosBit}(1, 0)\) to be
both False (for 2 = 010) and True (for 3 = 011) at the same time—a contradiction.

**Summary:**
- One-hot: AMO requires \(O(V^2)\) explicit pairwise clauses
- Bit encoding: AMO is guaranteed by binary arithmetic—**zero clauses needed**

The range restriction clauses then narrow the domain from \(\{0, \ldots, 2^b - 1\}\) to
\(\{0, \ldots, V - 1\}\), completing the "exactly one valid vertex" guarantee.

#### Range restriction clauses

At each time \(t\), we forbid codes 5, 6, 7. Consider code 5 = 101:

\[
\text{forbid\_code}(t, 5) \;=\;
\lnot\text{PosBit}(t, 0) \;\lor\; \text{PosBit}(t, 1) \;\lor\; \lnot\text{PosBit}(t, 2)
\]

This clause says: "at least one bit must differ from 101."

Similarly for code 6 = 110:

\[
\text{forbid\_code}(t, 6) \;=\;
\text{PosBit}(t, 0) \;\lor\; \lnot\text{PosBit}(t, 1) \;\lor\; \lnot\text{PosBit}(t, 2)
\]

And code 7 = 111:

\[
\text{forbid\_code}(t, 7) \;=\;
\lnot\text{PosBit}(t, 0) \;\lor\; \lnot\text{PosBit}(t, 1) \;\lor\; \lnot\text{PosBit}(t, 2)
\]

With \(T = 2\), there are 3 timesteps (\(t = 0, 1, 2\)), so \(3 \times 3 = 9\) range
restriction clauses total.

#### Transition constraint example

Suppose vertex 2 has neighbors \(\{1, 3, 4\}\), plus itself for waiting:
\(\text{adj}(2) = \{1, 2, 3, 4\}\).

The constraint at \(t = 0\) is:

\[
\text{pos}(0) = 2
\;\;\Longrightarrow\;\;
\text{pos}(1) = 1 \;\lor\; \text{pos}(1) = 2 \;\lor\; \text{pos}(1) = 3 \;\lor\; \text{pos}(1) = 4
\]

Expanding the left-hand side (\(2 = 010\)):

\[
\bigl(\lnot\text{PosBit}(0,0) \land \text{PosBit}(0,1) \land \lnot\text{PosBit}(0,2)\bigr)
\;\Longrightarrow\;
\text{(disjunction of successor codes)}
\]

When converted to CNF, this implication \(A \to B\) becomes \(\lnot A \lor B\), which
is then distributed over the bit-level conjunctions.

#### Start and goal fixing

To fix start = 0 and goal = 4:

**Start (\(s = 0 = 000\)):**

\[
\lnot\text{PosBit}(0, 0),\quad \lnot\text{PosBit}(0, 1),\quad \lnot\text{PosBit}(0, 2)
\]

**Goal (\(g = 4 = 100\)):**

\[
\lnot\text{PosBit}(2, 0),\quad \lnot\text{PosBit}(2, 1),\quad \text{PosBit}(2, 2)
\]

These are 6 unit clauses total.

---

### Correspondence to old encoding

The meaning is preserved, but encoded compositionally:

| Old variable       | New representation                                                    |
|--------------------|-----------------------------------------------------------------------|
| At(\(t\), \(v\))   | \(\bigwedge_k (\text{PosBit}(t,k) \leftrightarrow v_k)\)              |
| Move(\(t\), \(u\), \(v\)) | \((\text{pos}(t) = u) \land (\text{pos}(t+1) = v)\) for \(u \ne v\) |
| Wait(\(t\), \(v\)) | \((\text{pos}(t) = v) \land (\text{pos}(t+1) = v)\)                   |

We no longer have explicit Move/Wait variables; transitions are implicit in
consecutive position constraints.

### Scaling properties

For fixed horizon \(T\):

| Encoding           | Position variables    | Action variables         |
|--------------------|-----------------------|--------------------------|
| Old (one-hot)      | \(O(T \cdot V)\)      | \(O(T \cdot V^2)\) dense |
| New (bit)          | \(O(T \cdot \log V)\) | none (implicit)          |

This logarithmic reduction is why the bit encoding scales better as \(V\) grows.

### Edge cases

- **\(V = 1\)**: \(b = 1\); only code 0 is valid. No range restriction needed.
- **\(V\) power of two**: No invalid codes, so no range restriction clauses.
- **\(T = 0\)**: No transitions; SAT iff start = goal.
- **Decoding**: \(\text{path}[t] = \sum_k 2^k \cdot [\![\text{PosBit}(t,k)]\!]\),
  guaranteed \(< V\) when range clauses are satisfied.


# Acknowledgements

Many thanks for contributions over the years. I got bug reports, corrected code, and other support from Darius Bacon, Phil Ruggera, Peng Shao, Amit Patil, Ted Nienstedt, Jim Martin, Ben Catanzariti, and others. Now that the project is on GitHub, you can see the [contributors](https://github.com/aimacode/aima-python/graphs/contributors) who are doing a great job of actively improving the project. Many thanks to all contributors, especially [@darius](https://github.com/darius), [@SnShine](https://github.com/SnShine), [@reachtarunhere](https://github.com/reachtarunhere), [@antmarakis](https://github.com/antmarakis), [@Chipe1](https://github.com/Chipe1), [@ad71](https://github.com/ad71) and [@MariannaSpyrakou](https://github.com/MariannaSpyrakou).

<!---Reference Links-->
[agents]:../master/agents.py
[csp]:../master/csp.py
[games]:../master/games.py
[grid]:../master/grid.py
[knowledge]:../master/knowledge.py
[learning]:../master/learning.py
[logic]:../master/logic.py
[mdp]:../master/mdp.py
[nlp]:../master/nlp.py
[planning]:../master/planning.py
[probability]:../master/probability.py
[rl]:../master/rl.py
[search]:../master/search.py
[utils]:../master/utils.py
[text]:../master/text.py

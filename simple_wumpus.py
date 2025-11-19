from argparse import Action
from logic4e import (
    Expr,
    KB,
    PropKB,
    tt_entails,
    wumpus,
    pit,
    ok_to_move,
    breeze,
    stench,
    equiv,
    implies,
    new_disjunction,
    dpll_satisfiable,
    associate,
    WumpusPosition,
    Agent,
    Bump,
    Glitter,
    Stench,
    Breeze,
    Scream,
)
from agents4e import Bump, Glitter, Stench, Breeze, Scream
from search import PlanRoute, astar_search
from typing import *
from mat_sat import (
    mat_sat_cpp,
    mat_sat_mpspdz,
    mat_sat_mpspdz_async,
    reserve_ports_for_formula_sets,
)
import asyncio

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
CENTER = 4


def location(x, y):
    return Expr("Location", x, y)


def wumpus_alive():
    return Expr("WumpusAlive")


def bump(x, y):
    return Expr("Bump", x, y)


def ok_to_move(x, y):
    return Expr("OK", x, y)


def glitter(x, y):
    return Expr("Glitter", x, y)


## creating a simple hybrid wumpus agent that doesnt keep temporal data
class SimpleWumpusKB(PropKB):
    def __init__(self, dimrow):
        super().__init__()
        self.dimrow = dimrow
        self.tell(~wumpus(1, 1))
        self.tell(~pit(1, 1))
        for y in range(1, dimrow + 1):
            for x in range(1, dimrow + 1):
                for formula in self.make_ok_to_move_physics(x, y):
                    self.tell(formula)

    def make_ok_to_move_physics(self, x: int, y: int) -> List[Expr]:
        formulas = list()
        formulas.append(
            equiv(
                ok_to_move(x, y),
                ~pit(x, y) & ~bump(x, y) & (~wumpus(x, y) | ~wumpus_alive()),
            )
        )
        pits_in = list()
        wumpus_in = list()

        if x > 1:  # West room exists
            pits_in.append(pit(x - 1, y))
            wumpus_in.append(wumpus(x - 1, y))

        if y < self.dimrow:  # North room exists
            pits_in.append(pit(x, y + 1))
            wumpus_in.append(wumpus(x, y + 1))

        if x < self.dimrow:  # East room exists
            pits_in.append(pit(x + 1, y))
            wumpus_in.append(wumpus(x + 1, y))

        if y > 1:  # South room exists
            pits_in.append(pit(x, y - 1))
            wumpus_in.append(wumpus(x, y - 1))

        formulas.append(equiv(breeze(x, y), new_disjunction(pits_in)))
        formulas.append(equiv(stench(x, y), new_disjunction(wumpus_in)))
        formulas.append(location(x, y))
        return formulas

    def make_percept_sentence(
        self, directional_percepts: List[List[Any]], location: Tuple[int, int]
    ):
        cx, cy = location
        # the directions are: Left, Right, Up, Down, Center
        offsets = {
            LEFT: (-1, 0),
            RIGHT: (1, 0),
            UP: (0, 1),
            DOWN: (0, -1),
            CENTER: (0, 0),
        }
        # glitter,  stench, breeze, scream
        flags = [0, 0, 0]
        walls: set[Tuple[int, int]] = set()
        for i, percept_list in enumerate(directional_percepts):
            if len(percept_list) > 0:
                percept = percept_list[0]
                # treat the bumps as different, tell the kb the specific location of the bump
                if isinstance(percept, Bump):
                    dx, dy = offsets[i]
                    nx, ny = cx + dx, cy + dy
                    self.tell(bump(nx, ny))
                    walls.add((nx, ny))
                ## all other percepts just tell they were percepted from the current location
                else:
                    if isinstance(percept, Glitter):
                        flags[0] = 1
                        self.tell(glitter(cx, cy))
                    elif isinstance(percept, Stench):
                        flags[1] = 1
                        self.tell(stench(cx, cy))
                    elif isinstance(percept, Breeze):
                        flags[2] = 1
                        self.tell(breeze(cx, cy))
                    elif isinstance(percept, Scream):
                        self.tell(~wumpus_alive())
        # if didnt percieve bump in that direction mark not bump
        for dx, dy in offsets.values():
            if (cx + dx, cy + dy) not in walls:
                self.tell(~bump(cx + dx, cy + dy))
        for i in range(len(flags)):
            if flags[i] == 0:  # if not percepted
                if i == 0:
                    self.tell(~glitter(cx, cy))
                elif i == 1:
                    self.tell(~stench(cx, cy))
                elif i == 2:
                    self.tell(~breeze(cx, cy))

    def ask_if_true(self, query):
        formula = associate("&", list(self.clauses)) & ~query
        result = dpll_satisfiable(formula)  # if returns a model then false
        return False if result else True


class SimpleWumpusKBMatSat(SimpleWumpusKB):
    def ask_if_true(self, query):
        formula = associate("&", list(self.clauses)) & ~query
        result = mat_sat_cpp(formula)
        return False if result else True


class MultiAgentWumpusKB(KB):
    def __init__(self, dimrow: int, agents_location: List[Tuple[int, int]] = [(1, 1)]):
        self.dimrow = dimrow
        self.num_agents = len(agents_location)
        self.public_KB = PropKB()
        self.private_KB = [PropKB() for _ in range(len(agents_location))]

        for agent_id, l in enumerate(agents_location):
            self.tell(agent_id, ~wumpus(l[0], l[1]))
            self.tell(agent_id, ~pit(l[0], l[1]))

        for y in range(1, self.dimrow + 1):
            for x in range(1, self.dimrow + 1):
                for formula in self.make_ok_to_move_physics(x, y):
                    self.tell_public(formula)

    def make_ok_to_move_physics(self, x: int, y: int) -> List[Expr]:
        formulas = list()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        for dx, dy in directions:
            i, j = x + dx, y + dy
            if 1 <= i < self.dimrow and 1 <= j < self.dimrow:
                formulas.append(
                    equiv(
                        ok_to_move(i, j),
                        ~pit(i, j) & ~bump(i, j) & (~wumpus(i, j) | ~wumpus_alive()),
                    )
                )
                pits_in = list()
                wumpus_in = list()

                if i > 1:  # West room eiists
                    pits_in.append(pit(i - 1, j))
                    wumpus_in.append(wumpus(i - 1, j))

                if j < self.dimrow:  # North room eiists
                    pits_in.append(pit(i, j + 1))
                    wumpus_in.append(wumpus(i, j + 1))

                if i < self.dimrow:  # East room eiists
                    pits_in.append(pit(i + 1, j))
                    wumpus_in.append(wumpus(i + 1, j))

                if j > 1:  # South room eiists
                    pits_in.append(pit(i, j - 1))
                    wumpus_in.append(wumpus(i, j - 1))

                formulas.append(equiv(breeze(i, j), new_disjunction(pits_in)))
                formulas.append(equiv(stench(i, j), new_disjunction(wumpus_in)))
                formulas.append(location(i, j))
        return formulas

    def make_percept_sentence(
        self,
        agent_id: int,
        directional_percepts: List[List[Any]],
        location: Tuple[int, int],
    ):
        """
        Process percepts for a specific agent and update their PRIVATE KB.

        IMPORTANT: Each agent's percepts are stored in their own private KB!
        - Agent 0's percepts go to self.private_KB[0]
        - Agent 1's percepts go to self.private_KB[1]
        - etc.

        This ensures agents only reason about what THEY have observed.
        Shared knowledge (like Wumpus death) goes to public_KB via tell_public().
        """
        assert (
            0 <= agent_id < self.num_agents
        ), f"Invalid agent_id {agent_id}, must be 0-{self.num_agents-1}"
        cx, cy = location
        # the directions are: Left, Right, Up, Down, Center
        offsets = {
            LEFT: (-1, 0),
            RIGHT: (1, 0),
            UP: (0, 1),
            DOWN: (0, -1),
            CENTER: (0, 0),
        }
        # glitter,  stench, breeze, scream
        flags = [0, 0, 0]
        walls: set[Tuple[int, int]] = set()
        for i, percept_list in enumerate(directional_percepts):
            if len(percept_list) > 0:
                percept = percept_list[0]
                # treat the bumps as different, tell the kb the specific location of the bump
                if isinstance(percept, Bump):
                    dx, dy = offsets[i]
                    nx, ny = cx + dx, cy + dy
                    self.tell(agent_id, bump(nx, ny))
                    walls.add((nx, ny))
                ## all other percepts just tell they were percepted from the current location
                else:
                    if isinstance(percept, Glitter):
                        # flags[0] = 1
                        # self.tell(agent_id, glitter(cx, cy))
                        continue
                    elif isinstance(percept, Stench):
                        flags[1] = 1
                        self.tell(agent_id, stench(cx, cy))
                    elif isinstance(percept, Breeze):
                        flags[2] = 1
                        self.tell(agent_id, breeze(cx, cy))
                    elif isinstance(percept, Scream):
                        self.tell_public(~wumpus_alive())
        # if didnt percieve bump in that direction mark not bump
        for dx, dy in offsets.values():
            if (cx + dx, cy + dy) not in walls:
                self.tell(agent_id, ~bump(cx + dx, cy + dy))
        for i in range(len(flags)):
            if flags[i] == 0:  # if not percepted
                if i == 0:
                    # self.tell(agent_id, ~glitter(cx, cy))
                    continue
                elif i == 1:
                    self.tell(agent_id, ~stench(cx, cy))
                elif i == 2:
                    self.tell(agent_id, ~breeze(cx, cy))

    def ask_if_true(self, agent_id: int, query, secure=False):
        """
        Query using public KB and ALL agents' private KBs combined.

        TEMPORARY FOR DEMONSTRATION: This combines ALL private KBs so agents
        can share percepts and reason about the world together.

        Combines:
        1. Public KB (shared by all agents)
        2. ALL agents' private KBs (combined knowledge from all agents)
        """
        assert (
            0 <= agent_id < self.num_agents
        ), f"Invalid agent_id {agent_id}, must be 0-{self.num_agents-1}"

        # Combine public KB with ALL agents' private KB clauses (shared percepts)
        combined_clauses = list(self.public_KB.clauses)
        for agent_private_kb in self.private_KB:
            combined_clauses.extend(list(agent_private_kb.clauses))

        if not combined_clauses:
            return False
        formula = associate("&", combined_clauses) & ~query
        result = dpll_satisfiable(formula)  # if returns a model then false
        return False if result else True

    def ask_generator(self, query):
        """Yield the empty substitution {} if KB entails query; else no results."""
        all_clauses = list(self.public_KB.clauses) + [
            clause for sublist in self.private_KB for clause in sublist.clauses
        ]
        if tt_entails(Expr("&", *list(all_clauses)), query):
            yield {}

    def tell_public(self, sentence: Expr):
        """Add a sentence to the public KB (shared by all agents)"""
        self.public_KB.tell(sentence)

    def tell(self, agent_id: int, sentence: Expr):
        """Add a sentence to a specific agent's private KB"""
        self.private_KB[agent_id].tell(sentence)

    def get_kb_stats(self):
        """
        Get statistics about the knowledge base for debugging.
        Shows that each agent has their own private KB.
        """
        stats = {"public_clauses": len(list(self.public_KB.clauses)), "agents": []}
        for agent_id in range(self.num_agents):
            agent_stats = {
                "agent_id": agent_id,
                "private_clauses": len(list(self.private_KB[agent_id].clauses)),
            }
            stats["agents"].append(agent_stats)
        return stats

    def print_kb_stats(self):
        """Print KB statistics showing private percepts per agent"""
        stats = self.get_kb_stats()
        print("\n" + "=" * 50)
        print("KNOWLEDGE BASE STATISTICS")
        print("=" * 50)
        print(f"Public KB (shared): {stats['public_clauses']} clauses")
        print("\nPrivate KBs (agent-specific percepts):")
        for agent_info in stats["agents"]:
            print(
                f"  Agent {agent_info['agent_id']}: {agent_info['private_clauses']} clauses"
            )
        print("=" * 50)


class MultiAgentWumpusKBMatSatSecure(MultiAgentWumpusKB):
    """Version of MultiAgentWumpusKB that uses secure MatSat multi-party computation.

    Each agent's private knowledge is kept private, and queries are answered
    using secure multi-party computation where each agent contributes their
    private KB as a separate party.
    """

    def __init__(
        self,
        dimrow: int,
        agents_location: List[Tuple[int, int]] = [(1, 1)],
        batch_size: int | None = None,
        timeout: int = 300,
        sequential: bool = False,
    ):
        """
        Initialize the secure multi-party computation KB.

        Args:
            dimrow: Dimension of the wumpus world grid
            agents_location: List of starting locations for each agent
            batch_size: Default number of queries to process in each batch.
                       If None, processes all queries at once (default behavior).
            timeout: Timeout in seconds for MP-SPDZ execution (default: 300)
            sequential: If True, process queries sequentially by default (one at a time).
                       If False, process queries concurrently in batches (default).
        """
        super().__init__(dimrow, agents_location)
        self.batch_size = batch_size
        self.timeout = timeout
        self.sequential = sequential

    def _build_agent_formulas(self) -> List[Expr]:
        """Build formulas for MP-SPDZ where each agent contributes their private KB + public KB.

        Returns:
            List of formulas, one per agent (each formula becomes one party in MP-SPDZ)
        """
        public_clauses = list(self.public_KB.clauses)
        formulas = []

        # For each agent, create a formula: (public KB & agent's private KB)
        for agent_idx in range(self.num_agents):
            agent_private_clauses = list(self.private_KB[agent_idx].clauses)
            if agent_private_clauses:
                # Combine public and this agent's private clauses
                agent_clauses = public_clauses + agent_private_clauses
                agent_formula = associate("&", agent_clauses)
                formulas.append(agent_formula)
            elif public_clauses:
                # If agent has no private clauses, just use public KB
                formulas.append(associate("&", public_clauses))

        return formulas

    def _build_query_formulas(self, agent_id: int, query: Expr) -> List[Expr] | None:
        """Build formulas with query added for unsatisfiability check.

        Each agent's formula contains: public KB + their own private KB
        Only the querying agent's formula additionally contains: ~query

        Args:
            agent_id: The agent making the query
            query: The query expression to check

        Returns:
            List of formulas with ~query added to the querying agent's formula,
            or None if no formulas available
        """
        formulas = self._build_agent_formulas()
        if not formulas:
            return None

        # Validate agent_id is within bounds
        if agent_id < 0 or agent_id >= len(formulas):
            return None

        # Add ~query to check if (KB & ~query) is unsatisfiable
        # Only add ~query to the querying agent's formula (their private KB + public KB + ~query)
        formulas[agent_id] = formulas[agent_id] & ~query
        print(f"formulas: {formulas}")
        return formulas

    def _build_ok_to_move_query_formulas(
        self, agent_id: int, query: Expr, physics_clauses: List[Expr]
    ) -> List[Expr] | None:
        """Build formulas tailored for ok_to_move queries.

        Combines the shared physics for the target cell with:
          - Each agent's private KB clauses (agent-specific percepts)
          - Negates the query and adds it to the specified agent's formula
        """
        base_clauses = list(physics_clauses)

        if not base_clauses:
            return None

        formulas: List[Expr] = []
        for agent_idx in range(self.num_agents):
            agent_private = list(self.private_KB[agent_idx].clauses)
            combined = base_clauses + agent_private if agent_idx == agent_id else agent_private
            if combined:
                formulas.append(associate("&", combined))

        if not formulas:
            return None
        if agent_id < 0 or agent_id >= len(formulas):
            return None

        formulas[agent_id] = formulas[agent_id] & ~query
        return formulas

    def _validate_agent_id(self, agent_id: int) -> None:
        """Validate agent_id is within bounds."""
        assert (
            0 <= agent_id < self.num_agents
        ), f"Invalid agent_id {agent_id}, must be 0-{self.num_agents-1}"

    def ask_if_true(self, agent_id: int, query):
        """
        Query using secure MatSat solver with public KB and ALL agents' private KBs combined.

        Uses MP-SPDZ secure multi-party computation where:
        - Public KB is shared by all parties (included in each party's formula)
        - Each agent's private KB is a separate party
        - The query is checked for unsatisfiability (KB & ~query)

        Args:
            agent_id: The agent making the query
            query: The query expression to check

        Returns:
            True if KB entails query, False otherwise
        """
        self._validate_agent_id(agent_id)

        # Debug: Print query being processed
        print(f"\n[DEBUG] ask_if_true: Processing query for agent {agent_id}: {query}")

        formulas = self._build_query_formulas(agent_id, query)
        if not formulas:
            print(f"  [DEBUG] ask_if_true: No valid formula set, returning False")
            return False

        print(
            f"  [DEBUG] ask_if_true: Built {len(formulas)} formula(s), starting MP-SPDZ computation... (timeout={self.timeout}s)"
        )

        # Run secure multi-party MatSat computation
        # Each formula becomes one party in MP-SPDZ
        result = asyncio.run(mat_sat_mpspdz_async(formulas, timeout=self.timeout))

        # If result is None, the formula is unsatisfiable, meaning KB entails query
        # If result is a dict, the formula is satisfiable, meaning KB does not entail query
        is_entailed = result is None
        if isinstance(result, Exception):
            print(f"  [DEBUG] ask_if_true: ERROR - {result}, returning False")
        else:
            print(
                f"  [DEBUG] ask_if_true: Result = {is_entailed} (result={'None (UNSAT=entailed)' if result is None else 'SAT (not entailed)'})"
            )
        return is_entailed

    async def ask_if_true_async(self, agent_id: int, query):
        """
        Async version of ask_if_true for concurrent queries.

        Args:
            agent_id: The agent making the query
            query: The query expression to check

        Returns:
            True if KB entails query, False otherwise
        """
        self._validate_agent_id(agent_id)

        # Debug: Print query being processed
        print(
            f"\n[DEBUG] ask_if_true_async: Processing query for agent {agent_id}: {query}"
        )

        formulas = self._build_query_formulas(agent_id, query)
        if not formulas:
            print(f"  [DEBUG] ask_if_true_async: No valid formula set, returning False")
            return False

        print(
            f"  [DEBUG] ask_if_true_async: Built {len(formulas)} formula(s), starting MP-SPDZ computation... (timeout={self.timeout}s)"
        )

        # Run secure multi-party MatSat computation
        result = await mat_sat_mpspdz_async(formulas, timeout=self.timeout)

        # If result is None, the formula is unsatisfiable, meaning KB entails query
        # If result is a dict, the formula is satisfiable, meaning KB does not entail query
        is_entailed = result is None
        if isinstance(result, Exception):
            print(f"  [DEBUG] ask_if_true_async: ERROR - {result}, returning False")
        else:
            print(
                f"  [DEBUG] ask_if_true_async: Result = {is_entailed} (result={'None (UNSAT=entailed)' if result is None else 'SAT (not entailed)'})"
            )
        return is_entailed

    def ask_if_okay_to_move(self, agent_id: int, x: int, y: int) -> bool:
        """Optimized query that focuses on the ok_to_move physics for (x, y)."""
        self._validate_agent_id(agent_id)

        query = ok_to_move(x, y)
        physics = list(self.make_ok_to_move_physics(x, y))
        formulas = self._build_ok_to_move_query_formulas(agent_id, query, physics)
        if not formulas:
            print(
                f"[DEBUG] ask_if_okay_to_move: Unable to build formulas for agent {agent_id}, returning False"
            )
            return False

        print(
            f"\n[DEBUG] ask_if_okay_to_move: Agent {agent_id} checking OK({x}, {y}) with {len(physics)} physics clause(s)"
        )

        result = mat_sat_mpspdz(
            formulas, protocol="shamir", timeout=self.timeout, debug=True
        )
        is_entailed = result is None
        print(
            f"[DEBUG] ask_if_okay_to_move: Result = {is_entailed} (result={'None (UNSAT=entailed)' if result is None else 'SAT (not entailed)'})"
        )
        return is_entailed

    async def ask_if_okay_to_move_async(self, agent_id: int, x: int, y: int) -> bool:
        """Async variant of ask_if_okay_to_move for concurrent planners."""
        self._validate_agent_id(agent_id)

        query = ok_to_move(x, y)
        physics = list(self.make_ok_to_move_physics(x, y))
        formulas = self._build_ok_to_move_query_formulas(agent_id, query, physics)
        if not formulas:
            print(
                f"[DEBUG] ask_if_okay_to_move_async: Unable to build formulas for agent {agent_id}, returning False"
            )
            return False

        print(
            f"\n[DEBUG] ask_if_okay_to_move_async: Agent {agent_id} checking OK({x}, {y}) with {len(physics)} physics clause(s)"
        )

        result = await mat_sat_mpspdz_async(formulas, timeout=self.timeout)
        is_entailed = result is None
        print(
            f"[DEBUG] ask_if_okay_to_move_async: Result = {is_entailed} (result={'None (UNSAT=entailed)' if result is None else 'SAT (not entailed)'})"
        )
        return is_entailed

    async def ask_if_true_batch_async(
        self,
        agent_id: int,
        queries: List[Expr],
        batch_size: int | None = None,
        sequential: bool | None = None,
    ):
        """
        Batch version that processes multiple queries concurrently or sequentially.

        This method processes queries either:
        - Sequentially: One at a time using ask_if_true_async (sequential=True)
        - Concurrently: In batches using concurrent MP-SPDZ calls (sequential=False)

        Args:
            agent_id: The agent making the queries
            queries: List of query expressions to check
            batch_size: Number of queries to process in each batch (only used if sequential=False).
                       If None, uses the instance's batch_size attribute, or processes
                       all queries at once if that is also None.
            sequential: If True, process queries one at a time sequentially.
                       If False, process queries in batches concurrently.
                       If None, uses the instance's sequential attribute (default).

        Returns:
            List of boolean results, one for each query
        """
        self._validate_agent_id(agent_id)

        if not queries:
            return []

        # Use instance sequential setting if not explicitly provided
        if sequential is None:
            sequential = getattr(self, "sequential", False)

        # Sequential mode: process queries one at a time using ask_if_true_async
        if sequential:
            print(
                f"\n[DEBUG] ask_if_true_batch_async: Processing {len(queries)} queries SEQUENTIALLY (one at a time)"
            )
            results = []
            for i, query in enumerate(queries):
                print(f"\n[DEBUG] Sequential query {i+1}/{len(queries)}: {query}")
                result = await self.ask_if_true_async(agent_id, query)
                results.append(result)
            print(
                f"\n[DEBUG] ask_if_true_batch_async: Completed all {len(queries)} queries sequentially"
            )
            print(
                f"[DEBUG] Summary: {sum(results)}/{len(results)} queries returned True"
            )
            return results

        # Build formulas for each query
        # All queries in this batch are for the same agent (agent_id)
        all_formula_sets = []
        for query in queries:
            formulas = self._build_query_formulas(agent_id, query)
            all_formula_sets.append(formulas if formulas else [])

        if batch_size is None:
            batch_size = getattr(self, "batch_size", None)

        if batch_size is None or batch_size <= 0:
            batch_size = len(queries)

        # Debug: Print initial batch info
        num_batches = (len(queries) + batch_size - 1) // batch_size
        print(
            f"\n[DEBUG] ask_if_true_batch_async: Processing {len(queries)} queries in {num_batches} batch(es) (batch_size={batch_size})"
        )

        # Process queries in batches
        final_results = []
        batch_num = 0
        for batch_start in range(0, len(all_formula_sets), batch_size):
            batch_num += 1
            batch_end = min(batch_start + batch_size, len(all_formula_sets))
            batch_formula_sets = all_formula_sets[batch_start:batch_end]
            batch_queries = queries[batch_start:batch_end]

            # Debug: Print batch info and queries
            print(
                f"\n[DEBUG] Batch {batch_num}/{num_batches}: Processing queries {batch_start+1}-{batch_end} of {len(queries)}"
            )
            for i, query in enumerate(batch_queries):
                print(f"  Query {batch_start + i + 1}: {query}")

            # Reserve ports upfront for this batch to avoid race conditions
            valid_formula_sets = [fs for fs in batch_formula_sets if fs]
            if not valid_formula_sets:
                # All queries in this batch had no valid formula set
                print(
                    f"  [DEBUG] Batch {batch_num}: All queries had no valid formula set"
                )
                final_results.extend([False] * (batch_end - batch_start))
                continue

            reserved_ports = await reserve_ports_for_formula_sets(valid_formula_sets)
            print(
                f"  [DEBUG] Batch {batch_num}: Reserved {len(reserved_ports)} ports, starting concurrent processing... (timeout={self.timeout}s per query)"
            )

            # Run all queries in this batch concurrently with pre-reserved ports
            batch_results = await asyncio.gather(
                *[
                    mat_sat_mpspdz_async(formula_set, port=port, timeout=self.timeout)
                    for formula_set, port in zip(valid_formula_sets, reserved_ports)
                ],
                return_exceptions=True,
            )

            # Process results: None means unsatisfiable (KB entails query)
            # Map results back to original query list for this batch
            print(f"  [DEBUG] Batch {batch_num}: Results:")
            valid_idx = 0
            for i, formula_set in enumerate(batch_formula_sets):
                query_idx = batch_start + i
                if formula_set:
                    # This query had a valid formula set
                    result = batch_results[valid_idx]
                    is_entailed = (
                        result is None if not isinstance(result, Exception) else False
                    )
                    if isinstance(result, Exception):
                        print(
                            f"    Query {query_idx + 1} ({batch_queries[i]}): ERROR - {result}"
                        )
                    else:
                        print(
                            f"    Query {query_idx + 1} ({batch_queries[i]}): {is_entailed} (result={'None (UNSAT=entailed)' if result is None else 'SAT (not entailed)'})"
                        )
                    final_results.append(is_entailed)
                    valid_idx += 1
                else:
                    # This query had no valid formula set
                    print(
                        f"    Query {query_idx + 1} ({batch_queries[i]}): False (no valid formula set)"
                    )
                    final_results.append(False)

        # Debug: Print final summary
        print(
            f"\n[DEBUG] ask_if_true_batch_async: Completed all {num_batches} batch(es)"
        )
        print(
            f"[DEBUG] Summary: {sum(final_results)}/{len(final_results)} queries returned True"
        )
        return final_results


class MultiAgentWumpusKBMatSat(MultiAgentWumpusKB):
    """

    Version of MultiAgentWumpusKB that uses MatSat solver for faster inference.

    TEMPORARY FOR DEMONSTRATION: Currently configured to share ALL percepts
    between agents (combines all private KBs). This allows collaborative reasoning.
    """

    def ask_if_true(self, agent_id: int, query):
        """
        Query using MatSat solver with public KB and ALL agents' private KBs combined.

        TEMPORARY FOR DEMONSTRATION: This combines ALL private KBs so agents
        can share percepts and reason about the world together.

        Combines:
        1. Public KB (shared by all agents)
        2. ALL agents' private KBs (combined knowledge from all agents)

        Uses MatSat C++ solver for faster SAT solving compared to DPLL.
        """
        assert (
            0 <= agent_id < self.num_agents
        ), f"Invalid agent_id {agent_id}, must be 0-{self.num_agents-1}"

        # Debug: Print query being processed
        print(f"\n[DEBUG] ask_if_true: Processing query for agent {agent_id}: {query}")

        # Combine public KB with ALL agents' private KB clauses (shared percepts)
        combined_clauses = list(self.public_KB.clauses)
        for agent_private_kb in self.private_KB:
            combined_clauses.extend(list(agent_private_kb.clauses))

        if not combined_clauses:
            print(f"  [DEBUG] ask_if_true: No clauses available, returning False")
            return False

        print(
            f"  [DEBUG] ask_if_true: Built formula with {len(combined_clauses)} clause(s), starting MatSat C++ computation..."
        )

        formula = associate("&", combined_clauses) & ~query
        result = mat_sat_cpp(formula, debug=False)  # if returns a model then false
        is_entailed = False if result else True

        print(
            f"  [DEBUG] ask_if_true: Result = {is_entailed} (result={'None (UNSAT=entailed)' if result is None else 'SAT (not entailed)'})"
        )
        return is_entailed

    def ask_if_okay_to_move(self, agent_id, x, y):
        """optimized version of ask_if_true that only checks the ok_to_move physics"""
        assert (
            0 <= agent_id < self.num_agents
        ), f"Invalid agent_id {agent_id}, must be 0-{self.num_agents-1}"

        query = ok_to_move(x, y)
        physics = list(self.make_ok_to_move_physics(x, y))

        print(
            f"\n[DEBUG] ask_if_okay_to_move: Agent {agent_id} checking OK({x}, {y}) with {len(physics)} physics clause(s)"
        )

        combined_clauses = []
        for agent_private_kb in self.private_KB:
            combined_clauses.extend(list(agent_private_kb.clauses))
        combined_clauses.extend(physics)

        if not combined_clauses:
            print(
                f"  [DEBUG] ask_if_okay_to_move: No clauses available, returning False"
            )
            return False

        print(
            f"  [DEBUG] ask_if_okay_to_move: Built formula with {len(combined_clauses)} clause(s), starting MatSat C++ computation..."
        )

        formula = associate("&", combined_clauses) & ~query
        result = mat_sat_cpp(formula, debug=True)
        is_entailed = False if result else True

        print(
            f"  [DEBUG] ask_if_okay_to_move: Result = {is_entailed} (result={'None (UNSAT=entailed)' if result is None else 'SAT (not entailed)'})"
        )
        return is_entailed


class SimpleHybridWumpusAgent(Agent):
    def __init__(self, dimentions, kb_class=SimpleWumpusKB):
        ## make sure the kb is of instance of SimpleWumpusKB or SimpleWumpusKBMatSat
        assert isinstance(kb_class, type) and issubclass(kb_class, SimpleWumpusKB)
        self.dimrow = dimentions
        self.kb = kb_class(self.dimrow)
        self.plan = list()
        self.current_position = WumpusPosition(1, 1, "RIGHT")
        self.have_arrow = True
        self.visited: set[Tuple[int, int]] = set()
        super().__init__(self.execute)

    def execute(self, directional_percepts: List[List[Any]]):
        # Environment provides: [[<Bump>], [None], [<Bump>], [None], [None]]
        # the directions are: Left, Right, Up, Down, Center
        self.kb.make_percept_sentence(
            directional_percepts, self.current_position.get_location()
        )

        CurrX, CurrY = self.current_position.get_location()
        CurrOrientation = self.current_position.get_orientation()

        self.visited.add((CurrX, CurrY))

        safe_points = list()
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                if self.kb.ask_if_true(ok_to_move(i, j)):
                    safe_points.append([i, j])

        # check if we have glitter and can leave
        if self.kb.ask_if_true(glitter(CurrX, CurrY)):
            goals = list()
            goals.append([1, 1])
            self.plan.append("Grab")
            actions = self.plan_route(self.current_position, goals, safe_points)
            self.plan.extend(actions)
            self.plan.append("Climb")

        # if not gold is found explore
        safe_unvisited: List[Tuple[int, int]] = []
        if len(self.plan) == 0:
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    if (i, j) not in self.visited and [i, j] in safe_points:
                        safe_unvisited.append((i, j))
            # now that we have all the safe spots find narrow it down to the ones we haven't visited

            goal = [1, 1] if not safe_unvisited else safe_unvisited[0]
            route = self.plan_route(self.current_position, goal, safe_points)
            self.plan.extend(route)

        if len(self.plan) == 0 and self.have_arrow:
            possible_wumpus = list()
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    if not self.kb.ask_if_true(wumpus(i, j)):
                        possible_wumpus.append([i, j])
            self.plan.extend(
                self.plan_shot(self.current_position, possible_wumpus, safe_points)
            )
            self.have_arrow = False

        # lets climb out if we are in [1,1] with no plan and no unvisited safe points
        if CurrX == 1 and CurrY == 1 and len(self.plan) == 0 and not safe_unvisited:
            self.plan.append("Climb")

        if len(self.plan) > 0:
            action = self.plan[0]
            self.plan = self.plan[1:]

            # Update position based on the action taken
            self.update_position(action)
            print("KB believes the current position is", self.current_position)

            return action
        else:
            return "NoOp"  # No action available

    def update_position(self, action):
        """Update the agent's position based on the action taken"""
        x, y = self.current_position.get_location()
        ori = self.current_position.get_orientation()

        if action == "Forward":
            dx, dy = 0, 0
            if ori == "UP":
                dy = 1
            elif ori == "DOWN":
                dy = -1
            elif ori == "LEFT":
                dx = -1
            elif ori == "RIGHT":
                dx = 1

            nx, ny = x + dx, y + dy
            # Check if move is valid (within bounds)
            if 1 <= nx <= self.dimrow and 1 <= ny <= self.dimrow:
                self.current_position.set_location(nx, ny)
            # If bump, position stays the same (handled by environment)

        elif action == "TurnLeft":
            rot = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
            self.current_position.set_orientation(rot[ori])

        elif action == "TurnRight":
            rot = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
            self.current_position.set_orientation(rot[ori])
        # For other actions (Grab, Climb, Shoot), position doesn't change

    def plan_route(self, current, goals, allowed):
        problem = PlanRoute(current, goals, allowed, self.dimrow)
        solution = astar_search(problem)
        return solution.solution() if solution else []

    def plan_shot(self, current, goals, allowed):
        shooting_positions = set()

        for loc in goals:
            x = loc[0]
            y = loc[1]
            for i in range(1, self.dimrow + 1):
                if i < x:
                    shooting_positions.add(WumpusPosition(i, y, "EAST"))
                if i > x:
                    shooting_positions.add(WumpusPosition(i, y, "WEST"))
                if i < y:
                    shooting_positions.add(WumpusPosition(x, i, "NORTH"))
                if i > y:
                    shooting_positions.add(WumpusPosition(x, i, "SOUTH"))

        # Can't have a shooting position from any of the rooms the Wumpus could reside
        orientations = ["EAST", "WEST", "NORTH", "SOUTH"]
        for loc in goals:
            for orientation in orientations:
                position_to_remove = WumpusPosition(loc[0], loc[1], orientation)
                if position_to_remove in shooting_positions:
                    shooting_positions.remove(position_to_remove)

        actions = list()
        actions.extend(self.plan_route(current, shooting_positions, allowed))
        actions.append("Shoot")
        return actions


class MultiAgentHybridWumpusAgent(Agent):
    """Multi-agent version where each agent has an ID and shares a knowledge base"""

    def __init__(
        self,
        agent_id: int,
        shared_kb: MultiAgentWumpusKB,
        dimensions: int,
        start_location: Tuple[int, int] = (1, 1),
    ):
        self.agent_id = agent_id
        self.kb = shared_kb
        self.dimrow = dimensions
        self.plan = list()
        self.current_position = WumpusPosition(
            start_location[0], start_location[1], "RIGHT"
        )
        self.have_arrow = True
        self.visited: set[Tuple[int, int]] = set()
        super().__init__(self.execute)

    def execute(self, percept_data):
        """
        Execute method that accepts (agent_id, directional_percepts) tuple.

        The agent_id is used to ensure percepts are stored in the correct private KB.
        """
        # Unpack the percept data
        agent_id, directional_percepts = percept_data
        # check if glitter in directions perpcepts
        glitter_found = False
        for percept in directional_percepts:
            if percept and len(percept) > 0:
                for single_percept in percept:
                    if single_percept and isinstance(single_percept, Glitter):
                        glitter_found = True
                        break

        # CRITICAL VALIDATION: Verify this is the right agent
        assert (
            agent_id == self.agent_id
        ), f"Agent {self.agent_id} received percepts for agent {agent_id}"

        # Update KB with percepts for THIS SPECIFIC agent (goes to private_KB[agent_id])
        kb_clauses_before = len(list(self.kb.private_KB[self.agent_id].clauses))
        self.kb.make_percept_sentence(
            self.agent_id, directional_percepts, self.current_position.get_location()
        )
        kb_clauses_after = len(list(self.kb.private_KB[self.agent_id].clauses))

        # Debug: Show that percepts were added to THIS agent's private KB
        new_clauses = kb_clauses_after - kb_clauses_before
        if new_clauses > 0:
            print(
                f"    → Added {new_clauses} clauses to Agent {self.agent_id}'s PRIVATE KB"
            )

        CurrX, CurrY = self.current_position.get_location()
        CurrOrientation = self.current_position.get_orientation()

        self.visited.add((CurrX, CurrY))

        # Collect all queries for concurrent processing
        query_positions = [
            (i, j) for i in range(1, self.dimrow) for j in range(1, self.dimrow)
        ]
        queries = [ok_to_move(i, j) for (i, j) in query_positions]

        # Fallback to sequential processing
        safe_points = list()
        if hasattr(self.kb, "ask_if_okay_to_move_async,nm"):

            async def gather_okay_positions():
                tasks = [
                    self.kb.ask_if_okay_to_move_async(self.agent_id, i, j)
                    for (i, j) in query_positions
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                safe: List[List[int]] = []
                for (i, j), result in zip(query_positions, results):
                    if isinstance(result, Exception):
                        print(
                            f"[DEBUG] ask_if_okay_to_move_async error at ({i}, {j}): {result}"
                        )
                        continue
                    if result:
                        safe.append([i, j])
                return safe

            safe_points = asyncio.run(gather_okay_positions())
        else:
            for (i, j), query in zip(query_positions, queries):
                if hasattr(self.kb, "ask_if_okay_to_move"):
                    if self.kb.ask_if_okay_to_move(self.agent_id, i, j):
                        safe_points.append([i, j])
                else:
                    if self.kb.ask_if_true(self.agent_id, query):
                        safe_points.append([i, j])
        # check if we have glitter and can leave
        if glitter_found:
            goals = list()
            goals.append([1, 1])
            self.plan.append("Grab")
            actions = self.plan_route(self.current_position, goals, safe_points)
            self.plan.extend(actions)
            self.plan.append("Climb")

        # if not gold is found explore
        safe_unvisited: List[Tuple[int, int]] = []
        if len(self.plan) == 0:
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    if (i, j) not in self.visited and [i, j] in safe_points:
                        safe_unvisited.append((i, j))
            # now that we have all the safe spots find narrow it down to the ones we haven't visited

            goal = [1, 1] if not safe_unvisited else safe_unvisited[0]
            route = self.plan_route(self.current_position, goal, safe_points)
            self.plan.extend(route)

        if len(self.plan) == 0 and self.have_arrow:
            # Collect all wumpus queries for concurrent processing
            wumpus_queries = []
            wumpus_positions = []
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    wumpus_queries.append(~wumpus(i, j))
                    wumpus_positions.append((i, j))

            # Process all queries concurrently if using async KB
            if hasattr(self.kb, "ask_if_true_batch_async"):
                wumpus_results = asyncio.run(
                    self.kb.ask_if_true_batch_async(self.agent_id, wumpus_queries)
                )
                possible_wumpus = [
                    [i, j]
                    for (i, j), result in zip(wumpus_positions, wumpus_results)
                    if not result
                ]
            else:
                # Fallback to sequential processing
                possible_wumpus = list()
                for i in range(1, self.dimrow + 1):
                    for j in range(1, self.dimrow + 1):
                        if not self.kb.ask_if_true(self.agent_id, ~wumpus(i, j)):
                            possible_wumpus.append([i, j])
            self.plan.extend(
                self.plan_shot(self.current_position, possible_wumpus, safe_points)
            )
            self.have_arrow = False

        # lets climb out if we are in [1,1] with no plan and no unvisited safe points
        if CurrX == 1 and CurrY == 1 and len(self.plan) == 0 and not safe_unvisited:
            self.plan.append("Climb")

        if len(self.plan) > 0:
            action = self.plan[0]
            self.plan = self.plan[1:]

            # Update position based on the action taken
            self.update_position(action)
            print(
                f"Agent {self.agent_id} KB believes the current position is",
                self.current_position,
            )

            return action
        else:
            return "NoOp"  # No action available

    def update_position(self, action):
        """Update the agent's position based on the action taken"""
        x, y = self.current_position.get_location()
        ori = self.current_position.get_orientation()

        if action == "Forward":
            dx, dy = 0, 0
            if ori == "UP":
                dy = 1
            elif ori == "DOWN":
                dy = -1
            elif ori == "LEFT":
                dx = -1
            elif ori == "RIGHT":
                dx = 1

            nx, ny = x + dx, y + dy
            # Check if move is valid (within bounds)
            if 1 <= nx <= self.dimrow and 1 <= ny <= self.dimrow:
                self.current_position.set_location(nx, ny)
            # If bump, position stays the same (handled by environment)

        elif action == "TurnLeft":
            rot = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
            self.current_position.set_orientation(rot[ori])

        elif action == "TurnRight":
            rot = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
            self.current_position.set_orientation(rot[ori])
        # For other actions (Grab, Climb, Shoot), position doesn't change

    def plan_route(self, current, goals, allowed):
        problem = PlanRoute(current, goals, allowed, self.dimrow)
        solution = astar_search(problem)
        return solution.solution() if solution else []

    def plan_shot(self, current, goals, allowed):
        shooting_positions = set()

        for loc in goals:
            x = loc[0]
            y = loc[1]
            for i in range(1, self.dimrow + 1):
                if i < x:
                    shooting_positions.add(WumpusPosition(i, y, "EAST"))
                if i > x:
                    shooting_positions.add(WumpusPosition(i, y, "WEST"))
                if i < y:
                    shooting_positions.add(WumpusPosition(x, i, "NORTH"))
                if i > y:
                    shooting_positions.add(WumpusPosition(x, i, "SOUTH"))

        # Can't have a shooting position from any of the rooms the Wumpus could reside
        orientations = ["EAST", "WEST", "NORTH", "SOUTH"]
        for loc in goals:
            for orientation in orientations:
                position_to_remove = WumpusPosition(loc[0], loc[1], orientation)
                if position_to_remove in shooting_positions:
                    shooting_positions.remove(position_to_remove)

        actions = list()
        actions.extend(self.plan_route(current, shooting_positions, allowed))
        actions.append("Shoot")
        return actions

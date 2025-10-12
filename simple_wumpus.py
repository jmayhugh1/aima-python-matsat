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
from mat_sat import mat_sat_cpp

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
                self.tell(
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

                if y < dimrow:  # North room exists
                    pits_in.append(pit(x, y + 1))
                    wumpus_in.append(wumpus(x, y + 1))

                if x < dimrow:  # East room exists
                    pits_in.append(pit(x + 1, y))
                    wumpus_in.append(wumpus(x + 1, y))

                if y > 1:  # South room exists
                    pits_in.append(pit(x, y - 1))
                    wumpus_in.append(wumpus(x, y - 1))

                self.tell(equiv(breeze(x, y), new_disjunction(pits_in)))
                self.tell(equiv(stench(x, y), new_disjunction(wumpus_in)))
                self.tell(location(x, y))

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
            self.tell_public(~wumpus(l[0], l[1]))
            self.tell_public(~pit(l[0], l[1]))

        for y in range(1, dimrow + 1):
            for x in range(1, dimrow + 1):
                self.tell_public(
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

                if y < dimrow:  # North room exists
                    pits_in.append(pit(x, y + 1))
                    wumpus_in.append(wumpus(x, y + 1))

                if x < dimrow:  # East room exists
                    pits_in.append(pit(x + 1, y))
                    wumpus_in.append(wumpus(x + 1, y))

                if y > 1:  # South room exists
                    pits_in.append(pit(x, y - 1))
                    wumpus_in.append(wumpus(x, y - 1))

                self.tell_public(equiv(breeze(x, y), new_disjunction(pits_in)))
                self.tell_public(equiv(stench(x, y), new_disjunction(wumpus_in)))
                self.tell_public(location(x, y))

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
                        flags[0] = 1
                        self.tell(agent_id, glitter(cx, cy))
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
                    self.tell(agent_id, ~glitter(cx, cy))
                elif i == 1:
                    self.tell(agent_id, ~stench(cx, cy))
                elif i == 2:
                    self.tell(agent_id, ~breeze(cx, cy))

    def ask_if_true(self, agent_id: int, query):
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

        # Combine public KB with ALL agents' private KB clauses (shared percepts)
        combined_clauses = list(self.public_KB.clauses)
        for agent_private_kb in self.private_KB:
            combined_clauses.extend(list(agent_private_kb.clauses))

        if not combined_clauses:
            return False
        formula = associate("&", combined_clauses) & ~query
        result = mat_sat_cpp(formula)  # if returns a model then false
        return False if result else True


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

        safe_points = list()
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                if self.kb.ask_if_true(self.agent_id, ok_to_move(i, j)):
                    safe_points.append([i, j])

        # check if we have glitter and can leave
        if self.kb.ask_if_true(self.agent_id, glitter(CurrX, CurrY)):
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

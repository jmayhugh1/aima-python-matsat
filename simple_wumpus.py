from argparse import Action
from logic4e import (
    Expr,
    PropKB,
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

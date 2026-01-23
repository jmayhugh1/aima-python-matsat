from private_path_query_utils import (
    compile_private_path_query,
    join_computation,
    Grid,
    Path,
    Protocol,
)
from pprint import pprint
from agents import Thing, Agent
from logic4e import PropKB
from typing import List, Tuple
from functools import lru_cache
from math import isfinite, log, exp
from collections import deque
import asyncio


# ===============================================================================
# KNOWLEDGE BASES
# ===============================================================================


# ==============================================================================
# PLAYERS
# ==============================================================================


class BayesMap:
    """store log odds probability map of the current map size"""

    eps = 1e-12

    def __init__(self, size: int = None, p_init: float = 0.2, map=None):
        if map is not None:
            if size is not None:
                assert len(map) == size, "Provided size does not match the map size"
            size = len(map)
            # Convert probability values to log-odds
            map = [
                [
                    BayesMap.probability_to_log_odds(map[i][j])
                    for j in range(len(map[i]))
                ]
                for i in range(len(map))
            ]
        else:
            assert size is not None, "Either size or map must be provided"

        self.size = size
        self.p_init = p_init
        self.map = (
            map
            if map is not None
            else [
                [BayesMap.probability_to_log_odds(p_init) for _ in range(size)]
                for _ in range(size)
            ]
        )  # size x size grid

    @staticmethod
    def log_odds_to_probability(log_odds: float) -> float:
        # stable sigmoid
        if log_odds >= 0:
            z = exp(-log_odds)
            p = 1.0 / (1.0 + z)
        else:
            z = exp(log_odds)
            p = z / (1.0 + z)

        # clamp to avoid exactly 0/1 due to saturation
        return min(1.0 - BayesMap.eps, max(BayesMap.eps, p))

    @staticmethod
    def probability_to_log_odds(p: float) -> float:
        # clamp to avoid log(0)
        p = min(1 - BayesMap.eps, max(BayesMap.eps, p))
        return log(p / (1 - p))

    @staticmethod
    def compute_entropy_map(log_odds_map: List[List[float]]) -> List[List[float]]:
        entropy_map = []
        for row in log_odds_map:
            entropy_row = []
            for log_odds in row:
                p = BayesMap.log_odds_to_probability(log_odds)
                p = min(1.0 - BayesMap.eps, max(BayesMap.eps, p))

                entropy = -(p * log(p) + (1 - p) * log(1 - p))
                entropy_row.append(entropy)
            entropy_map.append(entropy_row)
        return entropy_map

    def find_highest_entropy_path(self, length: int) -> Tuple[Path, Tuple[int, int]]:
        max_entropy = float("-inf")
        max_entropy_path = None
        max_entropy_path_start = (-1, -1)
        for x in range(self.size):
            for y in range(self.size):
                score, path = self._find_highest_entropy_path((x, y), length)
                if score > max_entropy:
                    max_entropy = score
                    max_entropy_path = path
                    max_entropy_path_start = (x, y)
        return max_entropy_path, (max_entropy_path_start[0], max_entropy_path_start[1])

    def _find_highest_entropy_path(self, start: Tuple[int, int], length: int) -> Path:
        """Maximize sum of entropies along a path of exactly `length` moves (revisits allowed)."""
        entropy_map = BayesMap.compute_entropy_map(self.map)
        n = self.size

        # 4-neighborhood; switch to 8-neighborhood if you want diagonals
        DIRS: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < n and 0 <= y < n

        @lru_cache(maxsize=None)
        def dp(
            x: int, y: int, steps_left: int
        ) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
            """
            Returns:
              (best_future_entropy, best_moves_tuple)
            where best_moves_tuple is a tuple of (dx, dy) moves of length `steps_left`.
            """
            if steps_left == 0:
                return 0.0, ()

            best_score = float("-inf")
            best_moves: Tuple[Tuple[int, int], ...] = ()

            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny):
                    continue

                future_score, future_moves = dp(nx, ny, steps_left - 1)
                score = entropy_map[nx][ny] + future_score

                if score > best_score:
                    best_score = score
                    best_moves = ((dx, dy),) + future_moves

            # If start is boxed in (shouldn't happen on normal grids), fall back gracefully
            if best_score == float("-inf"):
                return 0.0, ()

            return best_score, best_moves

        sx, sy = start
        max_entropy, best_moves_tuple = dp(sx, sy, length)
        best_moves = list(best_moves_tuple)
        return max_entropy, Path(start, best_moves)

    def update_probabilities(self, path: Path, safe: bool):
        cells = list(path.iter_path_cells(path))

        if safe:
            # All visited cells are safe => hazard prob = 0 => log-odds = -inf
            for x, y in cells:
                self.map[x][y] = -float("inf")
            return

        # unsafe: at least one hazard on the visited set
        ps = []
        for x, y in cells:
            lo = self.map[x][y]
            p = (
                self.log_odds_to_probability(lo)
                if isfinite(lo)
                else (1.0 if lo > 0 else 0.0)
            )
            ps.append(p)

        # P(all safe)
        p_all_safe = 1.0
        for p in ps:
            p_all_safe *= 1.0 - p

        p_unsafe = 1.0 - p_all_safe
        # If p_unsafe is ~0, the observation is extremely surprising; avoid divide-by-zero
        p_unsafe = max(p_unsafe, BayesMap.eps)

        # Update each visited cell: p'_k = p_k / P(unsafe)
        for (x, y), p in zip(cells, ps):
            p_post = p / p_unsafe
            p_post = min(1.0 - BayesMap.eps, max(BayesMap.eps, p_post))
            self.map[x][y] = self.probability_to_log_odds(p_post)

    def __str__(self):
        """
        Print the map as probabilities with color:
        - Green = close to 0
        - Red   = close to 1
        """

        def colorize(p: float) -> str:
            # Clamp to [0, 1]
            p = max(0.0, min(1.0, p))

            # Linear red-green gradient
            r = int(255 * p)
            g = int(255 * (1 - p))
            b = 0

            return f"\033[38;2;{r};{g};{b}m{p:.3f}\033[0m"

        string = ""
        for row in self.map:
            prob_row = []
            for lo in row:
                if isfinite(lo):
                    p = self.log_odds_to_probability(lo)
                else:
                    p = 1.0 if lo > 0 else 0.0

                prob_row.append(colorize(p))

            string += " ".join(prob_row) + "\n"

        return string

    def check_viable_path(self, start=(0, 0), end=None) -> bool:
        if end is None:
            end = (self.size - 1, self.size - 1)
        # do bfs only selecting cells with prob of hazard == 0.0
        n = self.size
        queue = deque([start])
        visited = set()
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if (x, y) == end:
                return True
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    lo = self.map[nx][ny]
                    p = (
                        self.log_odds_to_probability(lo)
                        if isfinite(lo)
                        else (1.0 if lo > 0 else 0.0)
                    )
                    if p <= self.eps and (nx, ny) not in visited:
                        queue.append((nx, ny))

        return False


class Bob:
    """contains knowledge of some portion of the map"""

    def __init__(self, grid: Grid, p_init: float = 0.5):
        self.grid: Grid = grid  # Ground truth map consists of zeros and ones only
        self.bayes_map: BayesMap = BayesMap(size=grid.dim, p_init=0.5)  # belief map
        self.grid_size = grid.dim

    pass


class Alice:
    """wants to find a safe path through the map"""

    def __init__(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        path_lengths: int,
        grid_size: int,
        p_init: float = 0.5,
    ):
        self.start = start  # the start location
        self.goal = goal
        self.path_length = (
            path_lengths  # contains the consistent path lenfth for each query
        )
        self.grid_size = grid_size
        self.p_init = p_init
        self.bayes_map = BayesMap(size=grid_size, p_init=p_init)  # belief map

    @staticmethod
    def show_path(bobs: List[Bob], path: Path) -> None:
        """
        Display the path on the true grid using colors:
        - Gray  : unvisited cells
        - Green : visited cells NOT overlapping a hazard
        - Red   : visited cells overlapping a hazard
        """

        GRAY = "\033[90m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        dim = bobs[0].grid_size

        # Collect all hazard cells from all Bobs
        hazard_cells = set()
        for bob in bobs:
            for x in range(dim):
                for y in range(dim):
                    if bob.grid.grid[x][y] == 1:
                        hazard_cells.add((x, y))

        visited_cells = set(path.iter_path_cells(path))

        for x in range(dim):
            row = []
            for y in range(dim):
                if (x, y) in visited_cells:
                    if (x, y) in hazard_cells:
                        row.append(f"{RED}●{RESET}")
                    else:
                        row.append(f"{GREEN}●{RESET}")
                else:
                    row.append(f"{GRAY}.{RESET}")
            print(" ".join(row))

    async def run_computation(
        self,
        bobs: List[Bob],
        iterations: int = 1,
        protocol: Protocol = Protocol.SHAMIR,
        base_port: int = 5001,
        show_progress: bool = False,
    ) -> None:
        assert len(bobs) > 0, "At least one Bob is required"
        if protocol == Protocol.SHAMIR:
            assert len(bobs) >= 2, "At least two bobs are required for Shamir protocol"
        assert iterations > 0, "Number of iterations must be positive"

        # make sure that Bob has the same grid size as Alice
        for bob in bobs:
            assert (
                bob.grid_size == self.grid_size
            ), "Bob's grid size must match Alice's grid size"

        num_parties = len(bobs) + 1
        ok = await compile_private_path_query(
            num_parties, self.grid_size, self.path_length
        )
        assert ok, "SPDZ compile failed"

        if show_progress:
            print("Initial belief map:")
            print(self.bayes_map)
            print("Starting Communication")
            print(f"The start is {self.start}, the goal is {self.goal}")

        for i in range(iterations):

            # gather highest entropy path
            path, start = self.bayes_map.find_highest_entropy_path(self.path_length)

            assert isinstance(path, Path), "Expected path to be of type Path"
            assert isinstance(
                start, tuple
            ), "Expected start to be of type Tuple[int, int]"

            if show_progress:
                print(f"--- Iteration {i+1} ---")
                print(f"Queried Path")
                Alice.show_path(bobs, path)

            inputs = [path] + [bob.grid for bob in bobs]

            results = await asyncio.gather(
                *[
                    join_computation(
                        id=i,
                        num_parties=num_parties,
                        input=inputs[i],
                        port=base_port,
                    )
                    for i in range(num_parties)
                ]
            )
            final_result = results[0]

            # update everyones understanding of the map
            for bob in bobs:
                bob.bayes_map.update_probabilities(path, final_result)
            self.bayes_map.update_probabilities(path, final_result)

            if show_progress:
                # print out updated belief map
                print("Updated belief map:")
                print(self.bayes_map)
            if self.bayes_map.check_viable_path(self.start, self.goal):
                if show_progress:
                    print("A viable path exists!")
                return True

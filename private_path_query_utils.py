from dataclasses import dataclass
import os
from typing import Tuple, List, Literal, Protocol, Optional
from abc import ABC, abstractmethod
import pathlib
import csv
import subprocess
import asyncio
from enum import Enum, IntEnum
import sys
import shutil
import re
import json
from datetime import datetime
import numpy as np

# Resolve paths relative to THIS file, not the CWD
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_SPDZ_ROOT = (_THIS_DIR / "MP-SPDZ").resolve()  # if MP-SPDZ is next to this file
# If MP-SPDZ is one level up, use: (_THIS_DIR.parent / "MP-SPDZ").resolve()

if not _SPDZ_ROOT.exists():
    raise FileNotFoundError(f"MP-SPDZ not found at: {_SPDZ_ROOT}")

# Make MP-SPDZ importable in this Python process *if needed*
sys.path.insert(0, str(_SPDZ_ROOT))

# Also set env for subprocesses
os.environ["PYTHONPATH"] = str(_SPDZ_ROOT)

spdz_root = str(_SPDZ_ROOT)

# Default program (for backward compatibility)
program = "private_path_query"
program_path = str(_SPDZ_ROOT / "Programs" / "Source" / f"{program}.py")
find_safe_path_program_path = str(
    _SPDZ_ROOT / "Programs" / "Source" / "multiparty_matsat.py"
)
run_parties_path = str(_SPDZ_ROOT / "run-parties.py")
encodings_path = str((_THIS_DIR / "path-encodings").resolve())
_TRACE_ARTIFACTS_ROOT = (_THIS_DIR / "trace-artifacts").resolve()
_TRACE_CSV_DIR = (_TRACE_ARTIFACTS_ROOT / "csv").resolve()
_TRACE_GRAPHS_DIR = (_TRACE_ARTIFACTS_ROOT / "graphs").resolve()

# ===============================================================================
# TYPES AND ENUMS
# ==============================================================================
Spot = Literal[0, 1, 2]
GraphSpot = Literal[0, 1, 2]


class EdgeState(IntEnum):
    """
    NO_EDGE means no edge exists.
    BLOCKED means edge exists but is not traversable.
    TRAVERSABLE means edge exists and is traversable.
    UNKNOWN is a public-domain marker used only for candidate edge domains;
    it must not be used in Bob's private graph adjacency payload.
    """

    NO_EDGE = 0
    BLOCKED = 1
    TRAVERSABLE = 2
    UNKNOWN = 3


class Protocol(Enum):
    SHAMIR = "shamir"
    MASCOT = "mascot"


class ProgramName(Enum):
    # Deprecated: prefer FIND_SAFE_PATH for new path-finding logic. Private path query s basically the same as verifier and is useless
    PRIVATE_PATH_QUERY = "private_path_query"
    VERIFIER = "verifier"
    FIND_SAFE_PATH = "matsat"


# ===============================================================================
# BASE CLASSES
# ===============================================================================


class BasePath(ABC):
    """Base class for all path types."""

    @abstractmethod
    def __str__(self) -> str:
        """Serialize the path for MPC input."""
        pass


class Environment(ABC):
    """Base class for all environment types (Grid, Graph, etc.)."""

    @abstractmethod
    def __str__(self) -> str:
        """Serialize the environment for MPC input."""
        pass


# ===============================================================================
# PATH IMPLEMENTATIONS
# ===============================================================================


class Path(BasePath):
    def __init__(self, start: Tuple[int, int], moves: List[Tuple[int, int]]):
        self.start: Tuple[int, int] = start
        self.moves: List[Tuple[int, int]] = moves
        for dx, dy in moves:
            assert dx in (-1, 0, 1), "Move dx must be -1, 0, or 1"
            assert dy in (-1, 0, 1), "Move dy must be -1, 0, or 1"

    def iter_path_cells(self, path: "Path"):
        """Yield all (x,y) cells visited by the path, including the start."""
        x, y = path.start
        yield (x, y)
        for dx, dy in path.moves:
            x, y = x + dx, y + dy
            yield (x, y)

    def __str__(self) -> str:
        res = ""
        res += f"{str(self.start[0])}\n{self.start[1]}\n"
        for dx, dy in self.moves:
            res += f"{dx}\n{dy}\n"
        return res

    def pretty_str(self):
        res = f"Start: {self.start}\nMoves:\n"
        for i, (dx, dy) in enumerate(self.moves):
            res += f"  Step {i+1}: ({dx}, {dy}) \n"
        return res


# ===============================================================================
# ENVIRONMENT IMPLEMENTATIONS
# ===============================================================================


class Grid(Environment):
    def __init__(self, grid=List[List[Spot]]):
        self.grid = grid
        self.dim = len(grid)
        assert all(len(row) == self.dim for row in grid), "Grid must be square"
        assert all(
            spot in (0, 1, 2) for row in grid for spot in row
        ), "Grid spots must be 0, 1, or 2"

    def __str__(self) -> str:
        res = ""
        for row in self.grid:
            res += " ".join(str(spot) for spot in row) + "\n"
        return res.rstrip()  # Remove trailing newline


@dataclass
class Vertex:
    id: int

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def __le__(self, other):
        return self.id <= other.id


@dataclass
class Edge:
    vertex1: Vertex
    vertex2: Vertex
    state: EdgeState = EdgeState.TRAVERSABLE
    weight: float | None = (
        None  # Relative importance weight for this edge (None = use default)
    )

    def __hash__(self):
        return hash(sorted((self.vertex1.id, self.vertex2.id)))


def normalize_edge_weights(
    edges: List["Edge"], total_budget: float = 1.0, default_weight: float = 1.0
) -> List["Edge"]:
    """
    Normalize edge weights so they sum to total_budget.

    Edges with weight=None are treated as having default_weight before normalization.
    All weights are then scaled proportionally to sum to total_budget.

    Args:
        edges: List of Edge objects
        total_budget: Total weight budget to distribute (default 1.0)
        default_weight: Weight assigned to edges without explicit weight (default 1.0)

    Returns:
        New list of Edge objects with normalized weights that sum to total_budget

    Example:
        >>> edges = [
        ...     Edge(v0, v1, weight=0.5),  # Explicit: high priority
        ...     Edge(v1, v2, weight=0.3),  # Explicit: medium priority
        ...     Edge(v2, v3),              # Implicit: uses default_weight=1.0
        ... ]
        >>> normalized = normalize_edge_weights(edges, total_budget=1.0)
        # Weights before normalization: [0.5, 0.3, 1.0] -> sum=1.8
        # After: [0.278, 0.167, 0.556] -> sum=1.0
    """
    if not edges:
        return edges

    # Compute raw weights (use default for None)
    raw_weights = [e.weight if e.weight is not None else default_weight for e in edges]
    total_raw = sum(raw_weights)

    if total_raw <= 0:
        # Fallback to equal weights
        equal_weight = total_budget / len(edges)
        return [Edge(e.vertex1, e.vertex2, e.state, equal_weight) for e in edges]

    # Scale to budget
    scale = total_budget / total_raw
    return [
        Edge(e.vertex1, e.vertex2, e.state, raw_weights[i] * scale)
        for i, e in enumerate(edges)
    ]


class Graph(Environment):
    """
    Graph environment used by path-query logic and MPC input serialization.

    Args:
        vertices: Vertex objects whose ids define matrix indices (0..N-1).
        edges: Optional edge list. Each Edge carries an EdgeState and is treated as
            undirected in this constructor (state is mirrored for both directions).
        blocked_edges: Optional legacy list of blocked edges; each entry is written
            as EdgeState.BLOCKED for both directions.
        edge_states: Optional explicit NxN adjacency-state matrix where each cell is
            0 (no edge), 1 (blocked), or 2 (traversable). When provided, this takes
            precedence over edges/blocked_edges.
    """

    def __init__(
        self,
        vertices: List[Vertex],
        edges: Optional[List[Edge]] = None,
        blocked_edges: Optional[List[Edge]] = None,
        edge_states: Optional[List[List[GraphSpot]]] = None,
    ):
        self.vertices = vertices
        self.edges = edges or []
        self.blocked_edges = blocked_edges or []

        # all ids in vertices are unique
        assert len(vertices) == len(
            set(vertex.id for vertex in vertices)
        ), "Vertices must have unique ids"

        n = len(vertices)
        if edge_states is not None:
            assert len(edge_states) == n and all(
                len(row) == n for row in edge_states
            ), "edge_states must be an NxN matrix"
            assert all(
                val in (0, 1, 2) for row in edge_states for val in row
            ), "edge_states values must be 0 (none), 1 (blocked), or 2 (traversable)"
            self.adjacency_list = [row[:] for row in edge_states]
        else:
            # 0=no edge, 1=edge exists but blocked, 2=edge exists and traversable
            self.adjacency_list = [[0] * n for _ in range(n)]
            for edge in self.blocked_edges:
                self.adjacency_list[edge.vertex1.id][edge.vertex2.id] = int(
                    EdgeState.BLOCKED
                )
                self.adjacency_list[edge.vertex2.id][edge.vertex1.id] = int(
                    EdgeState.BLOCKED
                )
            for edge in self.edges:
                state_val = int(edge.state)
                if state_val == int(EdgeState.UNKNOWN):
                    raise ValueError(
                        "EdgeState.UNKNOWN is for public edge-domain metadata only, "
                        "not Graph adjacency state."
                    )
                i, j = edge.vertex1.id, edge.vertex2.id
                # If the same edge is listed multiple times, keep the most permissive state.
                self.adjacency_list[i][j] = max(self.adjacency_list[i][j], state_val)
                self.adjacency_list[j][i] = max(self.adjacency_list[j][i], state_val)

    def __str__(self) -> str:
        """Print out the adjacency list of the graph."""
        res = ""
        for i in range(len(self.adjacency_list)):
            for j in range(len(self.adjacency_list[i])):
                res += f"{self.adjacency_list[i][j]} "
            res += "\n"
        return res.rstrip()

    def to_directed_edges(self) -> List[Edge]:
        """Convert adjacency edge-states into directed Edge objects."""
        vertices = [Vertex(id=i) for i in range(len(self.adjacency_list))]
        edges: List[Edge] = []
        for u in range(len(self.adjacency_list)):
            for v in range(len(self.adjacency_list[u])):
                if u == v:
                    continue
                state = int(self.adjacency_list[u][v])
                if state == int(EdgeState.NO_EDGE):
                    continue
                edges.append(
                    Edge(
                        vertex1=vertices[u],
                        vertex2=vertices[v],
                        state=EdgeState(state),
                    )
                )
        return edges


class GraphPath(BasePath):
    """
    Planned graph path represented as an ordered edge sequence.

    Args:
        start: Starting vertex.
        moves: Ordered edges to traverse. The first edge must start at `start`, and
            every next edge must continue from the previous edge's destination.
    """

    def __init__(self, start: Vertex, moves: List[Edge]):
        self.start = start
        self.moves = moves
        if moves:
            # First edge must start from the start vertex
            assert (
                moves[0].vertex1.id == start.id
            ), "Start vertex must be the first vertex in the path"
            # Edges must be consecutive
            for i in range(len(moves) - 1):
                assert (
                    moves[i].vertex2.id == moves[i + 1].vertex1.id
                ), "Edges must be consecutive"
            self.end = moves[-1].vertex2
        else:
            self.end = start

    def __str__(self) -> str:
        """prints out the pairs of verteces that are connected by an edge"""
        res = ""
        for edge in self.moves:
            res += f"{edge.vertex1.id} {edge.vertex2.id}\n"
        return res.rstrip()


@dataclass
class ComputationResult:
    information_gain: float
    is_solved: bool
    satisfied_clauses: float | None = None
    u_vector: List[int] | None = None
    hazards_on_path: int | None = None


@dataclass
class PrivatePathInfo:
    """
    Shared FIND_SAFE_PATH configuration agreed upon by all parties ahead of runtime.
    """

    num_parties: int
    T: int
    V: int
    rows_per_id: List[int]
    use_weight_vector: bool = False
    use_edge_domain: bool = False
    edge_domain_edges: List[Edge] | None = None

    def __post_init__(self):
        if self.num_parties < 1:
            raise ValueError("num_parties must be >= 1")
        if self.T < 0:
            raise ValueError("T must be >= 0")
        if self.V < 1:
            raise ValueError("V must be >= 1")
        if len(self.rows_per_id) != self.num_parties:
            raise ValueError("rows_per_id length must match num_parties")
        if any(rows < 1 for rows in self.rows_per_id):
            raise ValueError("rows_per_id must contain positive values")
        if self.use_edge_domain and self.edge_domain_edges is None:
            raise ValueError(
                "use_edge_domain=True requires edge_domain_edges with UNKNOWN states"
            )
        if self.edge_domain_edges:
            for edge in self.edge_domain_edges:
                if edge.state != EdgeState.UNKNOWN:
                    raise ValueError(
                        "edge_domain_edges must use EdgeState.UNKNOWN to avoid leaking Bob states"
                    )
                for v in (edge.vertex1.id, edge.vertex2.id):
                    if not (0 <= v < self.V):
                        raise ValueError(
                            "edge_domain_edges vertex id out of range for V"
                        )

    def edge_domain_pairs(self) -> List[tuple[int, int]] | None:
        """Return deterministic directed pairs used for compact symbol domains."""
        if not self.use_edge_domain or not self.edge_domain_edges:
            return None
        from private_path_query_logic import directed_pairs_from_edges

        return directed_pairs_from_edges(self.edge_domain_edges, self.V)

    @classmethod
    def from_dict(cls, config: dict) -> "PrivatePathInfo":
        """Build from a dict (e.g., parsed JSON)."""
        V = int(config["V"])
        domain_edges_cfg = config.get("edge_domain_edges") or config.get(
            "edge_domain_pairs"
        )
        domain_edges: List[Edge] | None = None
        if domain_edges_cfg:
            domain_edges = []
            for entry in domain_edges_cfg:
                if isinstance(entry, dict):
                    u = int(entry["u"])
                    v = int(entry["v"])
                else:
                    u = int(entry[0])
                    v = int(entry[1])
                domain_edges.append(
                    Edge(
                        vertex1=Vertex(u),
                        vertex2=Vertex(v),
                        state=EdgeState.UNKNOWN,
                    )
                )
        return cls(
            num_parties=int(config["num_parties"]),
            T=int(config["T"]),
            V=V,
            rows_per_id=[int(v) for v in config["rows_per_id"]],
            use_weight_vector=bool(config.get("use_weight_vector", False)),
            use_edge_domain=bool(config.get("use_edge_domain", False)),
            edge_domain_edges=domain_edges,
        )

    @classmethod
    def from_json_file(cls, json_path: str) -> "PrivatePathInfo":
        """Load PrivatePathInfo from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls.from_dict(config)


# ==============================================================================
# Execution Logic
# =============================================================================


def parse_output(
    output: str, program_name: ProgramName = ProgramName.PRIVATE_PATH_QUERY
) -> ComputationResult:
    def _find_key_word(key_word: str) -> str:
        i = output.find(key_word)
        if i == -1:
            raise ValueError(f"Could not find {key_word} in output")
        start = i + len(key_word)
        end = output.find("\n", start)
        is_solved_str = output[start:end].strip()
        return is_solved_str

    # Preferred: parse standardized RESULT_* payload lines when present.
    m_is_solved = re.search(r"RESULT_IS_SOLVED\s*=\s*(\d+)", output)
    if m_is_solved:
        is_solved = m_is_solved.group(1) == "1"
        info_match = re.search(
            r"RESULT_INFORMATION_GAIN\s*=\s*([-+]?\d+(?:\.\d+)?)", output
        )
        if not info_match:
            info_match = re.search(
                r"information_gain\s*=\s*([-+]?\d+(?:\.\d+)?)", output
            )
        information_gain = float(info_match.group(1)) if info_match else 0.0
        sat_match = re.search(
            r"RESULT_SATISFIED_CLAUSES\s*=\s*([-+]?\d+(?:\.\d+)?)", output
        )
        satisfied_clauses = float(sat_match.group(1)) if sat_match else None
        hazard_match = re.search(r"RESULT_HAZARDS_ON_PATH\s*=\s*(\d+)", output)
        hazards_on_path = int(hazard_match.group(1)) if hazard_match else None
        u_pairs = re.findall(r"RESULT_U\[(\d+)\]\s*=\s*(-?\d+)", output)
        u_vector = (
            [v for _, v in sorted(((int(i), int(v)) for i, v in u_pairs))]
            if u_pairs
            else None
        )
        return ComputationResult(
            information_gain=information_gain,
            is_solved=is_solved,
            satisfied_clauses=satisfied_clauses,
            u_vector=u_vector,
            hazards_on_path=hazards_on_path,
        )

    if program_name == ProgramName.VERIFIER:
        # Verifier outputs: "Path is safe: %s" where %s is 1 (safe) or 0 (unsafe)
        # and "Hazards on path (count of matches): %s"
        safe_str = _find_key_word("Path is safe: ")
        is_solved = safe_str.strip() == "1"
        info_match = re.search(r"information_gain\s*=\s*([-+]?\d+(?:\.\d+)?)", output)
        information_gain = float(info_match.group(1)) if info_match else 0.0
        satisfied_clauses = None
        u_vector = None
        hazards_match = re.search(
            r"Hazards on path \(count of matches\):\s*(\d+)", output
        )
        hazards_on_path = int(hazards_match.group(1)) if hazards_match else None
    elif program_name == ProgramName.FIND_SAFE_PATH:
        # matsat outputs: "is_solved = %s" and "satisfied clauses = %s"
        match = re.search(r"is_solved\s*=\s*(\d+)", output)
        if not match:
            raise ValueError("Could not find is_solved in output")
        is_solved = match.group(1) == "1"
        information_gain = 0.0
        sat_match = re.search(r"satisfied clauses\s*=\s*([-+]?\d+(?:\.\d+)?)", output)
        satisfied_clauses = float(sat_match.group(1)) if sat_match else None
        u_pairs = re.findall(r"u\[(\d+)\]\s*=\s*(-?\d+)", output)
        u_vector = (
            [v for _, v in sorted(((int(i), int(v)) for i, v in u_pairs))]
            if u_pairs
            else None
        )
        hazards_on_path = None
    else:
        # private_path_query outputs: "information_gain=" and "is_solved="
        information_gain = float(_find_key_word("information_gain="))
        is_solved = _find_key_word("is_solved=") == "1"
        satisfied_clauses = None
        u_vector = None
        hazards_on_path = None

    return ComputationResult(
        information_gain=information_gain,
        is_solved=is_solved,
        satisfied_clauses=satisfied_clauses,
        u_vector=u_vector,
        hazards_on_path=hazards_on_path,
    )


def delete_persistence():
    """Delete the Persistence folder in MP-SPDZ directory."""
    persistence_dir = _SPDZ_ROOT / "Persistence"
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)
        print(f"Deleted Persistence folder: {persistence_dir}")
    else:
        print(f"Persistence folder does not exist: {persistence_dir}")


async def compile_private_path_query(
    num_parties: int,
    grid_size: int,
    query_size: int,
    iteration_no: int = 0,
    is_graph: bool = False,
) -> bool:
    """
    Compile the private_path_query MP-SPDZ program.

    Deprecated: prefer compile_find_safe_path() with ProgramName.FIND_SAFE_PATH
    for new path-finding logic.

    This function compiles the MPC program with the specified parameters. If
    iteration_no is 0, it deletes the Persistence folder to ensure a fresh start
    with default priors.

    Args:
        num_parties: Number of parties participating in the MPC computation.
        grid_size: Size of the grid (NxN) for the path query problem.
        query_size: Length of the path query.
        iteration_no: Iteration number (default: 0). If 0, deletes persistence
            to start with default priors.

    Returns:
        True if compilation succeeded, False otherwise.

    Raises:
        FileNotFoundError: If the MP-SPDZ program file is not found.
    """
    # Delete Persistence folder if starting from iteration 0, this causes the program to load a default prior
    if iteration_no == 0:
        delete_persistence()

    # compile the program
    path = [
        "python3",
        program_path,
        "--num_parties",
        str(num_parties),
        "--grid_size",
        str(grid_size),
        "--query_size",
        str(query_size),
        "--iteration_no",
        str(iteration_no),
    ]
    # Only add --is_graph flag if it's True (action="store_true" means flag presence = True)
    if is_graph:
        path.append("--is_graph")
    # Ensure we're using absolute path for cwd
    abs_spdz_root = pathlib.Path(_SPDZ_ROOT).resolve()

    process = await asyncio.create_subprocess_exec(
        *path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(abs_spdz_root),  # Run from MP-SPDZ directory
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print("Output:", stdout.decode())
        return True
    else:
        print("An error occurred while running the command.")
        print("Output:", stdout.decode())
        print("Errors:", stderr.decode())
        return False


async def compile_find_safe_path(
    num_parties: int | None = None,
    T: int | None = None,
    V: int | None = None,
    rows_per_party: int = 1,
    row_counts: List[int] | None = None,
    use_weight_vector: bool = False,
    weighted: bool | None = None,
    private_path_info: PrivatePathInfo | None = None,
) -> bool:
    """
    Compile the FIND_SAFE_PATH MP-SPDZ program (multiparty_matsat).

    Requires T and V in order to derive deterministic symbol count (num_vars)
    from private_path_query_logic.ordered_symbols(T, V).
    """
    if private_path_info is not None:
        num_parties = private_path_info.num_parties
        T = private_path_info.T
        V = private_path_info.V
        row_counts = private_path_info.rows_per_id
        use_weight_vector = private_path_info.use_weight_vector
    if weighted is not None:
        use_weight_vector = weighted
    if T is None or V is None:
        raise ValueError("T and V are required for compile_find_safe_path")
    if num_parties is None or num_parties < 1:
        raise ValueError("num_parties must be >= 1")

    # Local import to avoid circular dependency at module import time.
    from private_path_query_logic import ordered_symbols

    directed_pairs = (
        private_path_info.edge_domain_pairs() if private_path_info is not None else None
    )
    num_vars = len(ordered_symbols(T, V, directed_pairs=directed_pairs))

    if num_parties is None or num_parties < 1:
        raise ValueError("num_parties must be >= 1")
    path = [
        "python3",
        find_safe_path_program_path,
        "-F",
        "128",
        "--num_parties",
        str(num_parties),
        "--num_vars",
        str(num_vars),
    ]

    if row_counts is not None:
        if len(row_counts) != num_parties:
            raise ValueError("row_counts length must match num_parties")
        path.extend(["--row_counts", ",".join(str(v) for v in row_counts)])
    else:
        path.extend(["--rows_per_party", str(rows_per_party)])

    if use_weight_vector:
        path.append("--use_weight_vector")

    abs_spdz_root = pathlib.Path(_SPDZ_ROOT).resolve()
    process = await asyncio.create_subprocess_exec(
        *path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(abs_spdz_root),
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print("Output:", stdout.decode())
        return True
    else:
        print("An error occurred while running the command.")
        print("Output:", stdout.decode())
        print("Errors:", stderr.decode())
        return False


async def compile_verifier(
    num_parties: int,
    grid_size: int,
    query_size: int,
    is_graph: bool = False,
    iteration_no: int = 0,
) -> bool:
    """
    Compile the verifier MP-SPDZ program.

    This function compiles the verifier MPC program with the specified parameters.

    Args:
        num_parties: Number of parties participating in the MPC computation.
        grid_size: Size of the grid (NxN) for the path query problem.
        query_size: Length of the path query.
        is_graph: Whether to use graph mode (default: False).
        iteration_no: Iteration index for prior/posterior persistence (default: 0).

    Returns:
        True if compilation succeeded, False otherwise.

    Raises:
        FileNotFoundError: If the MP-SPDZ program file is not found.
    """
    verifier_program_path = str(_SPDZ_ROOT / "Programs" / "Source" / "verifier.py")

    # compile the program
    path = [
        "python3",
        verifier_program_path,
        "--num_parties",
        str(num_parties),
        "--grid_size",
        str(grid_size),
        "--query_size",
        str(query_size),
        "--iteration_no",
        str(iteration_no),
    ]
    # Only add --is_graph flag if it's True (action="store_true" means flag presence = True)
    if is_graph:
        path.append("--is_graph")
    # Ensure we're using absolute path for cwd
    abs_spdz_root = pathlib.Path(_SPDZ_ROOT).resolve()

    process = await asyncio.create_subprocess_exec(
        *path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(abs_spdz_root),  # Run from MP-SPDZ directory
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print("Output:", stdout.decode())
        return True
    else:
        print("An error occurred while running the command.")
        print("Output:", stdout.decode())
        print("Errors:", stderr.decode())
        return False


def _parse_matsat_trace_rows(output: str) -> List[dict]:
    """Parse solver iteration traces from stdout into structured rows."""
    try_re = re.compile(r"try_idx\s*=\s*(\d+)")
    iter_re = re.compile(r"iter_idx\s*=\s*(\d+)")
    jsat_re = re.compile(r"jsat\s*=\s*([-+]?\d+(?:\.\d+)?)")
    grad_re = re.compile(r"grad_sq\s*=\s*([-+]?\d+(?:\.\d+)?)")
    eps_re = re.compile(r"epsilon\s*=\s*([-+]?\d+(?:\.\d+)?)")
    alpha_re = re.compile(r"uncapped alpha\s*=\s*([-+]?\d+(?:\.\d+)?)")
    err_re = re.compile(r"err\s*=\s*([-+]?\d+(?:\.\d+)?)")
    unsat_re = re.compile(r"unsat_clauses\s*=\s*(\d+)")

    rows: List[dict] = []
    current_try = -1
    saw_explicit_try = False
    last_iter: Optional[int] = None
    current_row: Optional[dict] = None

    for raw in output.splitlines():
        line = raw.strip()
        m_try = try_re.search(line)
        if m_try:
            current_try = int(m_try.group(1))
            saw_explicit_try = True
            last_iter = None
            continue
        m_iter = iter_re.search(line)
        if m_iter:
            iter_idx = int(m_iter.group(1))
            if not saw_explicit_try:
                if last_iter is None or iter_idx <= last_iter:
                    current_try += 1
            elif last_iter is not None and iter_idx <= last_iter:
                current_try += 1
            last_iter = iter_idx
            current_row = {"try_idx": current_try, "iter_idx": iter_idx}
            rows.append(current_row)
            continue

        if current_row is None:
            continue

        m_jsat = jsat_re.search(line)
        if m_jsat:
            current_row.setdefault("jsat", float(m_jsat.group(1)))
            continue

        m_grad = grad_re.search(line)
        if m_grad:
            current_row["grad_sq"] = float(m_grad.group(1))
            continue

        m_eps = eps_re.search(line)
        if m_eps:
            current_row["epsilon"] = float(m_eps.group(1))
            continue

        m_alpha = alpha_re.search(line)
        if m_alpha:
            current_row["alpha"] = float(m_alpha.group(1))
            continue

        m_err = err_re.search(line)
        if m_err:
            current_row["err"] = float(m_err.group(1))
            continue

        m_unsat = unsat_re.search(line)
        if m_unsat:
            current_row["unsat_clauses"] = int(m_unsat.group(1))
            continue

    return rows


def _write_matsat_trace_artifacts(
    output: str, run_name: str, test_name: str | None = None
) -> None:
    """
    Write one CSV per run and one graph per try for MatSat trace output.

    If test_name is provided, artifacts are organized under that name;
    otherwise a timestamp-based folder is used.
    """
    rows = _parse_matsat_trace_rows(output)
    if not rows:
        return

    _TRACE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    folder_name = (
        test_name if test_name else datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    csv_path = (_TRACE_CSV_DIR / f"{run_name}_{folder_name}.csv").resolve()
    fieldnames = [
        "try_idx",
        "iter_idx",
        "jsat",
        "grad_sq",
        "epsilon",
        "alpha",
        "err",
        "unsat_clauses",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"trace csv: {csv_path}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    by_try: dict[int, List[dict]] = {}
    for row in rows:
        by_try.setdefault(int(row["try_idx"]), []).append(row)

    graph_dir = (_TRACE_GRAPHS_DIR / run_name / folder_name).resolve()
    graph_dir.mkdir(parents=True, exist_ok=True)
    for try_idx, try_rows in by_try.items():
        sorted_rows = sorted(try_rows, key=lambda r: int(r["iter_idx"]))
        x = [int(r["iter_idx"]) for r in sorted_rows]
        y_jsat = [float(r["jsat"]) if "jsat" in r else np.nan for r in sorted_rows]
        y_grad = [
            float(r["grad_sq"]) if "grad_sq" in r else np.nan for r in sorted_rows
        ]
        y_alpha = [float(r["alpha"]) if "alpha" in r else np.nan for r in sorted_rows]
        y_err = [float(r["err"]) if "err" in r else np.nan for r in sorted_rows]
        y_unsat = [
            float(r["unsat_clauses"]) if "unsat_clauses" in r else np.nan
            for r in sorted_rows
        ]

        fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
        axes[0].plot(x, y_jsat, marker="o", markersize=2, linewidth=1)
        axes[0].set_ylabel("jsat")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(x, y_grad, marker="o", markersize=2, linewidth=1)
        axes[1].set_ylabel("grad_sq")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(x, y_alpha, marker="o", markersize=2, linewidth=1)
        axes[2].set_ylabel("uncapped alpha")
        axes[2].grid(True, alpha=0.3)
        axes[3].plot(x, y_err, marker="o", markersize=2, linewidth=1)
        axes[3].set_ylabel("err")
        axes[3].grid(True, alpha=0.3)
        axes[4].plot(x, y_unsat, marker="o", markersize=2, linewidth=1)
        axes[4].set_xlabel("iteration")
        axes[4].set_ylabel("unsat_clauses")
        axes[4].grid(True, alpha=0.3)
        fig.suptitle(f"{run_name} try={try_idx}")
        fig.tight_layout()
        out_path = (graph_dir / f"try_{try_idx}.png").resolve()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"trace graph: {out_path}")


async def join_computation(
    id: int,
    num_parties: int,
    input: Environment | Path | str,
    port: int | None = None,
    host: str | None = None,
    protocol: Protocol = Protocol.SHAMIR,
    program_name: ProgramName = ProgramName.PRIVATE_PATH_QUERY,
    test_name: str | None = None,
) -> ComputationResult:
    """
    Join an MPC computation as a party and execute the specified program.

    This function runs the MP-SPDZ party executable with the provided input (either
    a Grid or Path) and returns the computation result containing information gain
    and whether the path was solved (safe).

    Args:
        id: Party ID (0 for Alice, 1..N for Bobs).
        num_parties: Total number of parties in the computation.
        input: Either a Grid/Graph (for Bob parties), Path/GraphPath (for Alice
            party), or raw string payload to provide as input to the MPC program.
        port: Base port number for network communication (default: None, uses MP-SPDZ
            default of 5000). MP-SPDZ will assign ports as base_port + party_id.
        host: Hostname where party 0 is running (default: None, uses localhost).
        protocol: MPC protocol to use (default: Protocol.SHAMIR).
        program_name: Name of the program to execute (default: ProgramName.PRIVATE_PATH_QUERY).

    Returns:
        ComputationResult containing:
            - information_gain: Float value representing the information gain
              from the Bayesian update (for private_path_query) or 0.0 (for verifier).
            - is_solved: Boolean indicating if the path query was solved (True
              means path is safe, False means unsafe/unsatisfiable).

    Raises:
        ValueError: If input is None or if Shamir protocol is used with less than
            3 parties.
        RuntimeError: If the MPC computation fails (non-zero exit code).

    Note:
        The function runs the party executable as a subprocess and communicates
        via stdin/stdout. The input is serialized as a string and sent to the
        program. The output is parsed to extract information_gain and is_solved
        values.
    """

    if not input:
        raise ValueError("Either grid or path must be provided")
    if protocol == Protocol.SHAMIR and num_parties < 3:
        raise ValueError("Shamir protocol requires at least 3 parties")

    party_exe = os.path.join(spdz_root, f"{protocol.value}-party.x")
    payload = str(input)

    args = [
        party_exe,
        "-N",
        str(num_parties),
        "-I",
        "-p",
        str(id),
    ]

    if port is not None:
        args.extend(["-pn", str(port)])

    if host:
        args.extend(["-h", host])

    args.append("-v")
    args.append(program_name.value)

    print(f"Running command: {' '.join(args)}")

    # Ensure we're using absolute path for cwd
    abs_spdz_root = pathlib.Path(spdz_root).resolve()
    print(f"Running from directory: {abs_spdz_root}")
    print(f"Player-Data will be created at: {abs_spdz_root / 'Player-Data'}")

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(
            abs_spdz_root
        ),  # Run from MP-SPDZ directory so Player-Data and Persistence are created there
        env={
            **os.environ,
            "PYTHONPATH": str(_SPDZ_ROOT),
            "DYLD_LIBRARY_PATH": f"{spdz_root}:{os.environ.get('DYLD_LIBRARY_PATH','')}",
            "LD_LIBRARY_PATH": f"{spdz_root}:{os.environ.get('LD_LIBRARY_PATH','')}",
        },
    )

    # concurrently write input and read output
    input_task = process.communicate(
        input=payload.encode()
    )  # communicate handles stdin
    # wait for finish
    stdout_data, stderr_data = await input_task

    # communicate returns bytes, so we can decode them here if we didn't use the streaming loop.
    # But wait, communicate() reads stdout/stderr until EOF.
    # If we want *live* streaming, we shouldn't use communicate for reading, only for writing?
    # Actually, communicate() buffers everything in memory.
    # To do live streaming + capture, it's safer to avoid communicate() or use it only if we don't care about live.
    # Given the previous hang, live streaming is preferred.

    # Rethinking implementation for safety/conciseness in this tool call:
    # Just use communicate and print the result *after* (if it finishes).
    # If it hangs, we won't see it.
    # BUT, the user wants to see it run.
    # AND I need the result.

    # Let's revert to capturing, but Print it immediately after capture (before parsing).

    if process.returncode == 0:
        out_str = stdout_data.decode()
        print(out_str)  # Print for debug visibility
        if program_name == ProgramName.FIND_SAFE_PATH:
            _write_matsat_trace_artifacts(
                out_str, run_name=program_name.value, test_name=test_name
            )
        result: ComputationResult = parse_output(out_str, program_name)
        return result
    else:
        err_str = stderr_data.decode()
        print(stdout_data.decode())
        print(err_str)
        raise RuntimeError(f"Computation failed: {err_str}")

    async def join():
        pass


def _q_matrix_to_payload(q: np.ndarray, weights: np.ndarray | None = None) -> str:
    """
    Serialize Q matrix rows to MP-SPDZ stdin payload.

    If `weights` is provided, append one scalar weight per row after all Q rows.
    """
    lines: List[str] = []
    for row in q:
        lines.append(" ".join(str(int(v)) for v in row))
    if weights is not None:
        lines.extend(str(float(v)) for v in weights)
    return "\n".join(lines) + ("\n" if lines else "")


async def join_computation_find_safe_path(
    id: int,
    private_path_info: PrivatePathInfo,
    start: Vertex | None = None,
    goal: Vertex | None = None,
    graph: Graph | None = None,
    port: int | None = None,
    host: str | None = None,
    protocol: Protocol = Protocol.SHAMIR,
    compile_program: bool = True,
    weighted: bool | None = None,
    test_name: str | None = None,
) -> ComputationResult:
    """
    Dedicated join path for FIND_SAFE_PATH (matsat).

    Party roles:
      - Alice (id=0): sends [q_physics ; q_alice] built from (start, goal, T, V)
      - Bob (id>0): sends q_bob built from (graph, T, V)
      - When weighted=True (or private_path_info.use_weight_vector=True), each
        party also appends one clause-weight per local Q row.
    """
    if private_path_info.T is None or private_path_info.V is None:
        raise ValueError("private_path_info must include T and V")
    T = private_path_info.T
    V = private_path_info.V
    num_parties = private_path_info.num_parties

    if id < 0 or id >= num_parties:
        raise ValueError("id must be in [0, num_parties)")
    if protocol == Protocol.SHAMIR and num_parties < 3:
        raise ValueError("Shamir protocol requires at least 3 parties")

    # Local import avoids circular import at module load time.
    from private_path_query_logic import (
        ordered_symbols,
        build_physics_q,
        build_bob_q,
        build_alice_q,
        compute_hard_clause_weight,
    )

    directed_pairs = private_path_info.edge_domain_pairs()
    syms = ordered_symbols(T, V, directed_pairs=directed_pairs)
    hard_weight = compute_hard_clause_weight(num_parties)
    q_physics, w_physics = build_physics_q(
        T,
        V,
        symbols=syms,
        directed_pairs=directed_pairs,
        hard_clause_weight=hard_weight,
    )
    effective_weighted = (
        private_path_info.use_weight_vector if weighted is None else weighted
    )

    if id == 0:
        if start is None or goal is None:
            raise ValueError("Alice party (id=0) requires start and goal")
        q_alice, w_alice = build_alice_q(
            private_path_info, start=start.id, goal=goal.id, symbols=syms
        )
        q_party = np.concatenate([q_physics, q_alice], axis=0)
        w_party = np.concatenate([w_physics, w_alice], axis=0)
    else:
        if graph is None:
            raise ValueError("Bob parties (id>0) require graph input")
        bob_edges = graph.to_directed_edges()
        q_bob, w_bob = build_bob_q(private_path_info, edges=bob_edges, symbols=syms)
        q_party = q_bob
        w_party = w_bob

    if len(w_party) != q_party.shape[0]:
        raise ValueError("Per-party clause weights must match local row count")

    if compile_program and id == 0:
        ok = await compile_find_safe_path(
            private_path_info=private_path_info,
            weighted=effective_weighted,
        )
        if not ok:
            raise RuntimeError("FIND_SAFE_PATH compile failed")

    payload = _q_matrix_to_payload(
        q_party, weights=w_party if effective_weighted else None
    )
    return await join_computation(
        id=id,
        num_parties=num_parties,
        input=payload,
        port=port,
        host=host,
        protocol=protocol,
        program_name=ProgramName.FIND_SAFE_PATH,
        test_name=test_name,
    )

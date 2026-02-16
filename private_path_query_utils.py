from dataclasses import dataclass
import os
from typing import Tuple, List, Literal, Protocol, Optional
from abc import ABC, abstractmethod
import pathlib
import subprocess
import asyncio
from enum import Enum, IntEnum
import sys
import shutil
import re
import json
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

# ===============================================================================
# TYPES AND ENUMS
# ==============================================================================
Spot = Literal[0, 1, 2]
GraphSpot = Literal[0, 1, 2]


class EdgeState(IntEnum):
    """
    NO_EDGE means will never become traversable.
    BLOCKED means MAY become traversable.
    TRAVERSABLE means can be traversed.
    """

    NO_EDGE = 0
    BLOCKED = 1
    TRAVERSABLE = 2


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

    def __hash__(self):
        return hash(sorted((self.vertex1.id, self.vertex2.id)))


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

    @classmethod
    def from_dict(cls, config: dict) -> "PrivatePathInfo":
        """Build from a dict (e.g., parsed JSON)."""
        return cls(
            num_parties=int(config["num_parties"]),
            T=int(config["T"]),
            V=int(config["V"]),
            rows_per_id=[int(v) for v in config["rows_per_id"]],
            use_weight_vector=bool(config.get("use_weight_vector", False)),
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
    if T is None or V is None:
        raise ValueError("T and V are required for compile_find_safe_path")
    if num_parties is None or num_parties < 1:
        raise ValueError("num_parties must be >= 1")

    # Local import to avoid circular dependency at module import time.
    from private_path_query_logic import ordered_symbols

    num_vars = len(ordered_symbols(T, V))

    if num_parties is None or num_parties < 1:
        raise ValueError("num_parties must be >= 1")
    path = [
        "python3",
        find_safe_path_program_path,
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


async def join_computation(
    id: int,
    num_parties: int,
    input: Environment | Path | str,
    port: int | None = None,
    host: str | None = None,
    protocol: Protocol = Protocol.SHAMIR,
    program_name: ProgramName = ProgramName.PRIVATE_PATH_QUERY,
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
        result: ComputationResult = parse_output(out_str, program_name)
        return result
    else:
        err_str = stderr_data.decode()
        print(stdout_data.decode())
        print(err_str)
        raise RuntimeError(f"Computation failed: {err_str}")

    async def join():
        pass


def _q_matrix_to_payload(q: np.ndarray) -> str:
    """Serialize integer Q matrix rows to MP-SPDZ stdin payload."""
    lines: List[str] = []
    for row in q:
        lines.append(" ".join(str(int(v)) for v in row))
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
) -> ComputationResult:
    """
    Dedicated join path for FIND_SAFE_PATH (matsat).

    Party roles:
      - Alice (id=0): sends q_alice built from (start, goal, T, V)
      - Bob (id>0): sends [q_physics ; q_bob] built from (graph, T, V)
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
    )

    syms = ordered_symbols(T, V)
    q_physics = build_physics_q(T, V, symbols=syms)

    if id == 0:
        if start is None or goal is None:
            raise ValueError("Alice party (id=0) requires start and goal")
        q_party = build_alice_q(start.id, goal.id, T, V, symbols=syms)
    else:
        if graph is None:
            raise ValueError("Bob parties (id>0) require graph input")
        bob_edges = graph.to_directed_edges()
        q_bob = build_bob_q(bob_edges, T, V, symbols=syms)
        q_party = np.concatenate([q_physics, q_bob], axis=0)

    if compile_program and id == 0:
        ok = await compile_find_safe_path(
            private_path_info=private_path_info,
        )
        if not ok:
            raise RuntimeError("FIND_SAFE_PATH compile failed")

    payload = _q_matrix_to_payload(q_party)
    return await join_computation(
        id=id,
        num_parties=num_parties,
        input=payload,
        port=port,
        host=host,
        protocol=protocol,
        program_name=ProgramName.FIND_SAFE_PATH,
    )

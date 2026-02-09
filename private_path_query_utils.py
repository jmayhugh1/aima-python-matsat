from dataclasses import dataclass
import os
from typing import Tuple, List, Literal
import pathlib
import subprocess
import asyncio
from enum import Enum
import sys
import shutil

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

program = "private_path_query"
program_path = str(_SPDZ_ROOT / "Programs" / "Source" / f"{program}.py")
run_parties_path = str(_SPDZ_ROOT / "run-parties.py")
encodings_path = str((_THIS_DIR / "path-encodings").resolve())

# ===============================================================================
# TYPES AND ENUMS
# ==============================================================================
Spot = Literal[0, 1]


class Protocol(Enum):
    SHAMIR = "shamir"
    MASCOT = "mascot"


class Path:
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

    def __str__(self):
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


class Grid:
    def __init__(self, grid=List[List[Spot]]):
        self.grid = grid
        self.dim = len(grid)
        assert all(len(row) == self.dim for row in grid), "Grid must be square"
        assert all(
            spot in (0, 1) for row in grid for spot in row
        ), "Grid spots must be 0 or 1"

    def __str__(self):
        res = ""
        for row in self.grid:
            res += " ".join(str(spot) for spot in row) + "\n"
        return res.rstrip()  # Remove trailing newline


@dataclass
class ComputationResult:
    information_gain: float
    is_solved: bool


# ==============================================================================
# Execution Logic
# =============================================================================


def parse_output(output: str) -> ComputationResult:
    def _find_key_word(key_word: str) -> str:
        i = output.find(key_word)
        if i == -1:
            raise ValueError(f"Could not find {key_word} in output")
        start = i + len(key_word)
        end = output.find("\n", start)
        is_solved_str = output[start:end].strip()
        return is_solved_str

    information_gain = float(_find_key_word("information_gain="))
    is_solved = _find_key_word("is_solved=") == "1"
    return ComputationResult(information_gain, is_solved)


def delete_persistence():
    """Delete the Persistence folder in MP-SPDZ directory."""
    persistence_dir = _SPDZ_ROOT / "Persistence"
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)
        print(f"Deleted Persistence folder: {persistence_dir}")
    else:
        print(f"Persistence folder does not exist: {persistence_dir}")


async def compile_private_path_query(
    num_parties: int, grid_size: int, query_size: int, iteration_no: int = 0
) -> bool:
    """
    Compile the private_path_query MP-SPDZ program.

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
    input: Grid | Path,
    port: int | None = None,
    host: str | None = None,
    protocol: Protocol = Protocol.SHAMIR,
) -> ComputationResult:
    """
    Join an MPC computation as a party and execute the private_path_query program.

    This function runs the MP-SPDZ party executable with the provided input (either
    a Grid or Path) and returns the computation result containing information gain
    and whether the path was solved (safe).

    Args:
        id: Party ID (0 for Alice, 1..N for Bobs).
        num_parties: Total number of parties in the computation.
        input: Either a Grid (for Bob parties) or Path (for Alice party) to provide
            as input to the MPC program.
        port: Base port number for network communication (default: None, uses MP-SPDZ
            default of 5000). MP-SPDZ will assign ports as base_port + party_id.
        host: Hostname where party 0 is running (default: None, uses localhost).
        protocol: MPC protocol to use (default: Protocol.SHAMIR).

    Returns:
        ComputationResult containing:
            - information_gain: Float value representing the information gain
              from the Bayesian update.
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
    args.append(program)

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

    # Helper to read stream and print/capture
    cached_stdout = []
    cached_stderr = []

    async def read_stream(stream, cache, prefix=""):
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode()
            print(f"{prefix}{decoded}", end="")
            cache.append(decoded)

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
        result: ComputationResult = parse_output(out_str)
        return result
    else:
        err_str = stderr_data.decode()
        print(stdout_data.decode())
        print(err_str)
        raise RuntimeError(f"Computation failed: {err_str}")

    async def join():
        pass

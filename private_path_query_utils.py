import os
from typing import Tuple, List, Literal
from pathlib import Path
import subprocess
import asyncio
from enum import Enum
import sys

# Resolve paths relative to THIS file, not the CWD
_THIS_DIR = Path(__file__).resolve().parent
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

    def iter_path_cells(self, path: Path):
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
            for spot in row:
                res += f"{spot}\n"
        return res


# ==============================================================================
# Execution Logic
# =============================================================================


def parse_output(output: str) -> bool:
    key_word = "is_solved ="
    i = output.find(key_word)
    if i == -1:
        raise ValueError("Could not find 'is_solved' in output")
    start = i + len(key_word)
    end = output.find("\n", start)
    is_solved_str = output[start:end].strip()
    return is_solved_str == "1"


async def compile_private_path_query(num_parties: int, grid_size: int, query_size: int):
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
    ]
    process = await asyncio.create_subprocess_exec(
        *path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
) -> bool:
    """player join the computation on its own thread, need an id for bob"""

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

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
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
            print(f"{prefix}{decoded}", end='')
            cache.append(decoded)

    # concurrently write input and read output
    input_task = process.communicate(input=payload.encode()) # communicate handles stdin
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
        print(out_str) # Print for debug visibility
        return parse_output(out_str)
    else:
        err_str = stderr_data.decode()
        print(stdout_data.decode())
        print(err_str)
        raise RuntimeError(f"Computation failed: {err_str}")

    async def join():
        pass

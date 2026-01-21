import os
from typing import Tuple, List, Literal
from pathlib import Path
import subprocess
import asyncio
from enum import Enum

# Set the PYTHONPATH environment variable
spdz_root = os.path.abspath("./MP-SPDZ")
os.environ["PYTHONPATH"] = "./MP-SPDZ"

program = "private_path_query"
program_path = os.path.join(
    os.environ["PYTHONPATH"], "Programs", "Source", f"{program}.py"
)
run_parties_path = os.path.join(os.environ["PYTHONPATH"], "run-parties.py")
encodings_path = os.path.join("path-encodings")

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
    port: int,
    protocol: Protocol = Protocol.SHAMIR,
) -> bool:
    """player join the compuation on its own thread, need an id for bob"""

    if not input:
        raise ValueError("Either grid or path must be provided")
    if protocol == Protocol.SHAMIR and num_parties < 3:
        raise ValueError("Shamir protocol requires at least 3 parties")

    party_exe = os.path.join(spdz_root, f"{protocol.value}-party.x")
    payload = str(input)

    process = await asyncio.create_subprocess_exec(
        party_exe,
        "-N",
        str(num_parties),
        "-I",
        "-p",
        str(id),
        "-pn",
        str(port),
        program,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        env={
            **os.environ,
            "DYLD_LIBRARY_PATH": f"{spdz_root}:{os.environ.get('DYLD_LIBRARY_PATH','')}",
            "LD_LIBRARY_PATH": f"{spdz_root}:{os.environ.get('LD_LIBRARY_PATH','')}",
        },
    )
    stdout, stderr = await process.communicate(input=payload.encode())
    if process.returncode == 0:
        return parse_output(stdout.decode())
    else:
        raise RuntimeError(f"Computation failed: {stderr.decode()}")

    async def join():
        pass

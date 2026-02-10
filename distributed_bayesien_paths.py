import argparse
import asyncio
import sys
import os
import pickle
import struct
from typing import List, Tuple
from private_path_query_utils import ComputationResult

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bayesien_paths import Alice, Bob, BayesMap
from private_path_query_utils import (
    Grid,
    Path,
    Protocol,
    ComputationResult,
    compile_private_path_query,
    join_computation,
    spdz_root,
)
import re
import pathlib
from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich import box

console = Console(force_terminal=True)

def print_grid_rich(grid_data, title="Grid"):
    """
    Pretty print a grid using Rich.
    1s are Red (Hazard), 0s are Green (Safe).
    """
    # Simply mapping data to colored text
    # Assuming grid_data is list of lists
    
    # Check if it's a Grid object or raw list
    if hasattr(grid_data, "grid"):
        grid = grid_data.grid
    else:
        grid = grid_data
        
    rows = len(grid)
    cols = len(grid[0])
    
    # Use a simpler table or even just text to avoid heavy borders
    console.print(f"[bold underline]{title}[/bold underline]")
    
    for r in range(rows):
        line = ""
        for c in range(cols):
            val = grid[r][c]
            if val == 1:
                line += "[red]██[/red]" # Solid block for wall
            else:
                line += "[green]··[/green]" # Dots for empty space
        console.print(line)

def print_belief_rich(bayes_map, title="Belief Map"):
    """
    Print belief map heat-map style.
    """
    # Use console rule for separator
    console.rule(f"[cyan]{title}[/cyan]")
    # Print the map string directly but maybe highlight?
    # For now just print it, assuming the map's own __str__ is good enough
    print(str(bayes_map))
# ==============================================================================
# NETWORKING UTILS
# ==============================================================================

async def send_msg(writer, msg: object):
    """Pickle and send message with length prefix."""
    data = pickle.dumps(msg)
    length = len(data)
    writer.write(struct.pack('!I', length))
    writer.write(data)
    await writer.drain()

async def recv_msg(reader) -> object:
    """Receive length prefix and then pickled message."""
    length_bytes = await reader.readexactly(4)
    length = struct.unpack('!I', length_bytes)[0]
    data = await reader.readexactly(length)
    return pickle.loads(data)

# ==============================================================================
# ALICE LOGIC
# ==============================================================================


# ==============================================================================
# DISTRIBUTED CLASSES (Overrides)
# ==============================================================================

class DistributedBayesMap(BayesMap):
    """
    Distributed version of BayesMap that implements smarter path planning
    to avoid immediate backtracking (A->B->A).
    """
    def _find_highest_entropy_path(self, start: Tuple[int, int], length: int) -> Tuple[float, Path]:
        """Maximize sum of entropies along a path of exactly `length` moves."""
        entropy_map = BayesMap.compute_entropy_map(self.map)
        n = self.size
        DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < n and 0 <= y < n
            
        # We need lru_cache for DP efficiently
        # But we cannot decorate a nested function easily if we want 'self' access or closure
        # Standard approach: define it inside.
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(
            x: int, y: int, px: int, py: int, steps_left: int
        ) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
            """
            Returns: (best_future_entropy, best_moves_tuple)
            px, py are coordinates of PREVIOUS cell to avoid stepping back.
            Use -1, -1 for start.
            """
            if steps_left == 0:
                return 0.0, ()

            best_score = float("-inf")
            best_moves: Tuple[Tuple[int, int], ...] = ()

            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                
                # Bounds check
                if not in_bounds(nx, ny):
                    continue
                
                # Constraint: Do not step back to previous cell
                if (nx, ny) == (px, py):
                    continue
                
                # Check 2: Prune Known Hazards
                # The 'map' stores log-odds.
                # log-odds > 2.0 corresponds to probability > ~0.88
                # If we are fairly certain this cell is a wall, don't walk into it.
                # This breaks loops like (1,1)->(1,0) where (1,0) is known unsafe but adjacent to current.
                # We want to force the path to navigate AROUND the wall.
                if self.map[nx][ny] > 2.0:
                    continue

                future_score, future_moves = dp(nx, ny, x, y, steps_left - 1)
                
                # Current logic: Sum entropy of visited cells (even if revisited later in path)
                score = entropy_map[nx][ny] + future_score

                if score > best_score:
                    best_score = score
                    best_moves = ((dx, dy),) + future_moves

            if best_score == float("-inf"):
                return 0.0, ()

            return best_score, best_moves

        sx, sy = start
        
        # Optimization: maximize information gain by skipping known-safe/unsafe starts
        # REASONING:
        # If the start node has low entropy (is effectively known Safe or Unsafe), starting a path here
        # contributes ~0 to the total Information Gain score.
        # This causes ties with other paths (e.g., Unsafe->Unknown vs Unknown->Unknown).
        # Since the search order is deterministic (row-major), Alice often picks the first path found,
        # which starts at the known-unsafe "gateway" (like (1,0)).
        # Because the path starts at a wall/unsafe node, the query returns "UNSAFE" immediately, Alice learns nothing,
        # and repeats the cycle indefinitely.
        # By skipping low-entropy starts (entropy < 0.01, approx 99% certainty),
        # we force Alice to "jump" to the frontier of uncertainty.
        current_entropy = entropy_map[sx][sy]
        if current_entropy < 1e-1:
            # Return -inf so this path is not chosen by the outer loop
            return float("-inf"), Path(start, [])
            
        # Pass -1, -1 as dummy previous coordinates for start
        max_entropy, best_moves_tuple = dp(sx, sy, -1, -1, length)
        best_moves = list(best_moves_tuple)
        return max_entropy, Path(start, best_moves)

class DistributedAlice(Alice):
    """
    Distributed Alice that uses DistributedBayesMap.
    """
    def __init__(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        path_lengths: int,
        grid_size: int,
        p_init: float = 0.5,
    ):
        super().__init__(start, goal, path_lengths, grid_size, p_init)
        # Override the bayes_map with our distributed version
        # (Re-initialize it with the correct size and p_init)
        self.bayes_map = DistributedBayesMap(size=grid_size, p_init=p_init)
        
        # Assume start is safe (prob = 0 => log-odds = -inf)
        self.bayes_map.map[start[0]][start[1]] = -float("inf")


async def run_alice(args):
    console.rule(f"[bold blue]Starting Alice (Player {args.party_id}) on {args.host}:{args.port}...[/bold blue]")
    
    start = (0, 0)
    goal = (args.grid_size - 1, args.grid_size - 1)
    
    # Use DistributedAlice
    alice = DistributedAlice(start=start, goal=goal, path_lengths=args.path_length, grid_size=args.grid_size)
    protocol = Protocol(args.protocol)
    
    # 1. Start TCP Server for Path Sharing
    # We use the COMMON args.port + 100 so Bob knows where to connect
    # (args.base_port might differ if it includes party_id)
    path_server_port = args.port + 100
    bob_writers = []
    bob_readers = []
    
    connection_event = asyncio.Event()
    expected_bobs = args.num_parties - 1 # Parties - Alice
    
    async def handle_bob(reader, writer):
        addr = writer.get_extra_info('peername')
        print(f"Bob connected from {addr}")
        bob_writers.append(writer)
        bob_readers.append(reader)
        if len(bob_writers) == expected_bobs:
            connection_event.set()
            
    # Use '0.0.0.0' to listen on all interfaces (avoids localhost IPv4/v6 ambiguity)
    listen_host = '0.0.0.0' if args.host == 'localhost' else args.host
    server = await asyncio.start_server(handle_bob, listen_host, path_server_port)
    print(f"Alice listening for path sharing on {args.host}:{path_server_port}")
    
    async with server:
        print(f"Waiting for {expected_bobs} Bobs to connect...")
        await connection_event.wait()
        print("All Bobs connected!")
        
        # Compile MPC program if needed (Alice usually handles this in tests)
        print("Compiling MPC program...")
        ok = await compile_private_path_query(args.num_parties, args.grid_size, args.path_length)
        if not ok:
            print("MPC Compilation failed!")
            return

        print_belief_rich(alice.bayes_map, "Initial Belief Map")
 
        for i in range(args.iterations):
            console.print(f"\n[bold magenta]--- Iteration {i+1} ---[/bold magenta]")
            
            # 1. Plan path
            path, _ = alice.bayes_map.find_highest_entropy_path(alice.path_length)
            print(f"Queried Path: {path.pretty_str()}")
            
            # 2. Share path with Bobs
            print("Broadcasting path to Bobs...")
            broadcast_tasks = [send_msg(w, "START") for w in bob_writers]
            await asyncio.gather(*broadcast_tasks)
            
            # 3. Run MPC
            with console.status("[yellow]Running MPC computation...[/yellow]", spinner="dots"):
                # Use local safe function to avoid parsing errors in utils
                result = await safe_join_computation(
                    id=args.party_id,
                    num_parties=args.num_parties,
                    input=path,
                    port=args.base_port,
                    host=args.host,
                    protocol=protocol
                )
            is_safe = result.is_solved
            
            # 4. Calculate Information Gain with Belief Update (Alice's side)
            prev_entropy = alice.bayes_map.get_total_entropy()
            alice.bayes_map.update_probabilities(path, is_safe)
            new_entropy = alice.bayes_map.get_total_entropy()
            info_gain = prev_entropy - new_entropy
            
            status_color = "green" if is_safe else "red"
            status_text = "SAFE" if is_safe else "UNSAFE"
            console.print(f"MPC Result: [bold {status_color}]{status_text}[/bold {status_color}]")
            console.print(f"Info Gain: [bold cyan]{info_gain:.4f}[/bold cyan]")
            
            # 5. Share Info Gain with Bobs
            console.print("Broadcasting Info Gain to Bobs...")
            broadcast_info_tasks = [send_msg(w, info_gain) for w in bob_writers]
            await asyncio.gather(*broadcast_info_tasks)
            
            # 6. Synchronization Barrier: Wait for ACK from all Bobs
            # This prevents Alice from running ahead (e.g. Iter 60 while Bob is at 30)
            # console.print("Waiting for ACK from Bobs...", style="dim")
            ack_tasks = [recv_msg(r) for r in bob_readers]
            await asyncio.gather(*ack_tasks)
            
            # 6. Visualize
            
            # 5. Visualize
            print_belief_rich(alice.bayes_map, "Updated Belief Map")
            
            if alice.bayes_map.check_viable_path(alice.start, alice.goal):
                console.print("[bold green]A viable path exists![/bold green]")
                # Theoretically send 'Stop' signal or proceed, 
                # but for this loop we'll just finish iterations or exit.
                # For cleaner exit we should probably tell Bobs to stop, but for now we run fixed iterations.
                # Let's just break locally and maybe Bobs handle connection close or we just run all iters.
                # To keep it simple: run all iterations.
    
    console.print("[bold blue]Alice finished.[/bold blue]")

# ==============================================================================
# BOB LOGIC
# ==============================================================================

import random

def generate_partitioned_maze(grid_size: int, seed: int, party_id: int, num_bobs: int) -> Grid:
    """
    Generates a deterministic global maze using 'seed', then returns
    the subset of walls assigned to 'party_id'.
    Uses Recursive Backtracker to ensure a perfect maze (every cell reachable).
    """
    # 1. Deterministic Generation
    rng = random.Random(seed)
    
    # Initialize full grid with Walls (1)
    map_data = [[1] * grid_size for _ in range(grid_size)]
    
    # Helper to check bounds
    def in_bounds(r, c):
        return 0 <= r < grid_size and 0 <= c < grid_size

    # Carve start and goal immediately
    map_data[0][0] = 0
    map_data[grid_size-1][grid_size-1] = 0
    
    # Recursive Backtracker
    # stack of (r, c)
    stack = [(0, 0)]
    visited = set([(0, 0)])
    
    while stack:
        current_r, current_c = stack[-1]
        
        # Find unvisited neighbors (distance 2)
        neighbors = []
        for dr, dc in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nr, nc = current_r + dr, current_c + dc
            if in_bounds(nr, nc) and (nr, nc) not in visited:
                neighbors.append((nr, nc))
        
        if neighbors:
            # Choose random neighbor
            next_r, next_c = rng.choice(neighbors)
            
            # Carve path to neighbor
            # 1. Carve the wall in between
            wall_r, wall_c = (current_r + next_r) // 2, (current_c + next_c) // 2
            map_data[wall_r][wall_c] = 0
            # 2. Carve the neighbor itself
            map_data[next_r][next_c] = 0
            
            visited.add((next_r, next_c))
            stack.append((next_r, next_c))
        else:
            stack.pop()
            
    # Post-processing: Make sure Goal is reachable if grid_size is even
    # (Recursive backtracker on odd grid visits all odd cells. If even, last cell might be wall)
    # Simple fix: Carve neighbors of goal if they are walls to ensure connectivity
    # effectively opening the goal to the maze
    goal_r, goal_c = grid_size - 1, grid_size - 1
    if map_data[goal_r][goal_c] == 0: # It is open, check if connected
        # Check if any neighbor is 0
        connected = False
        for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nr, nc = goal_r + dr, goal_c + dc
            if in_bounds(nr, nc) and map_data[nr][nc] == 0:
                connected = True
                break
        if not connected:
            # Force connect to a valid neighbor (preferable one that is part of the maze)
            # Just carve the Left or Top neighbor
            if in_bounds(goal_r, goal_c - 1): map_data[goal_r][goal_c - 1] = 0
            if in_bounds(goal_r - 1, goal_c): map_data[goal_r - 1][goal_c] = 0

    # 2. Partitioning
    # My Grid will be all 0s (safe) EXCEPT for the walls (1s) that belong to ME.
    my_grid_data = [[0] * grid_size for _ in range(grid_size)]
    
    # Alice (Party 0) is not a Bob. Bobs are 1..N.
    # We need to map Bob party_ids to [0, num_bobs-1] for partitioning.
    # party_id is 1-based index (1, 2, ..., num_bobs)
    # owner_idx will be party_id - 1
    my_owner_idx = party_id - 1
    
    for r in range(grid_size):
        for c in range(grid_size):
            if map_data[r][c] == 1: # It is a wall
                # Who owns this wall?
                # Simple hash: (r * grid_size + c) % num_bobs
                owner = (r * grid_size + c) % num_bobs
                
                if owner == my_owner_idx:
                    my_grid_data[r][c] = 1
                else:
                    my_grid_data[r][c] = 0 # Someone else's wall, to me it's "safe" (empty space)
                    
    return Grid(my_grid_data)

def get_bob_grid(args) -> Grid:
    """
    Returns the grid for Bob.
    Priority 1: Seeded Maze Generation
    Priority 2: Hardcoded 3x3 map (if size=3)
    Priority 3: Default center hazard
    """
    if args.seed is not None:
        # Bobs are parties 1 to num_parties-1.
        # But 'num_parties' in args DOES INCLUDE Alice.
        # So number of Bobs = args.num_parties - 1.
        num_bobs = args.num_parties - 1
        return generate_partitioned_maze(args.grid_size, args.seed, args.party_id, num_bobs)

    if args.grid_size == 3:
        # User requested hardcoded map:
        grid_data = [
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0]
        ]
        return Grid(grid_data)
    
    # Fallback to old logic
    grid_data = [[0] * args.grid_size for _ in range(args.grid_size)]
    mid = args.grid_size // 2
    grid_data[mid][mid] = 1 
    if args.grid_size > 2:
        grid_data[1][0] = 1 
    return Grid(grid_data)

async def safe_join_computation(
    id: int,
    num_parties: int,
    input: Grid | Path,
    port: int | None = None,
    host: str | None = None,
    protocol: Protocol = Protocol.SHAMIR,
) -> ComputationResult:
    """
    Local implementation of join_computation that handles parsing more robustly.
    Does not crash if information_gain is missing.
    """
    program = "private_path_query"
    
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

    abs_spdz_root = pathlib.Path(spdz_root).resolve()
    print(f"Running from directory: {abs_spdz_root}")
    print(f"Player-Data will be created at: {abs_spdz_root / 'Player-Data'}")

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(abs_spdz_root),
        env={
            **os.environ,
            "PYTHONPATH": str(abs_spdz_root),
            "DYLD_LIBRARY_PATH": f"{spdz_root}:{os.environ.get('DYLD_LIBRARY_PATH','')}",
            "LD_LIBRARY_PATH": f"{spdz_root}:{os.environ.get('LD_LIBRARY_PATH','')}",
        },
    )

    input_task = process.communicate(input=payload.encode())
    stdout_data, stderr_data = await input_task

    if process.returncode == 0:
        out_str = stdout_data.decode()
        print(out_str)
        
        # Robust parsing logic
        
        # 1. Parse Information Gain (optional)
        info_gain = 0.0
        try:
            # Look for information_gain=VALUE or information_gain = VALUE
            match = re.search(r"information_gain\s*=\s*(\S+)", out_str)
            if match:
                info_gain = float(match.group(1))
        except Exception as e:
            print(f"Warning: Could not parse information_gain: {e}")
            info_gain = 0.0
            
        # 2. Parse Is Solved (robust)
        is_safe = False
        try:
            # Look for is_solved=VALUE or is_solved = VALUE
            match = re.search(r"is_solved\s*=\s*(\d)", out_str)
            if match:
                is_safe = (match.group(1) == "1")
            else:
                # Fallback: check stricty if exact match fails
                if "is_solved=1" in out_str or "is_solved = 1" in out_str:
                    is_safe = True
                elif "is_solved=0" in out_str or "is_solved = 0" in out_str:
                    is_safe = False
                else:
                    raise ValueError("Could not find is_solved status")
        except Exception as e:
            print(f"Error parsing is_solved: {e}")
            # In a real scenario, might want to raise, but for now log
            raise
            
        return ComputationResult(info_gain, is_safe)
    else:
        err_str = stderr_data.decode()
        print(stdout_data.decode())
        print(err_str)
        raise RuntimeError(f"Computation failed: {err_str}")


async def run_bob(args):
    console.rule(f"[bold green]Starting Bob (Player {args.party_id})...[/bold green]")
    
    # 0. Setup
    # Generate a random grid for Bob
    bob_grid = get_bob_grid(args)
    bob = Bob(bob_grid)
    protocol = Protocol(args.protocol)
    
    print_grid_rich(bob.grid, title=f"Obstacle Grid (1=Hazard, 0=Safe)")
    
    # Connect to Alice
    console.print(f"Connecting to Alice at {args.host}:{args.port}...")
    
    total_info_gain = 0.0

    
    # 1. Connect to Alice for Path Sharing
    # Use common args.port + 100
    path_server_port = args.port + 100
    host = args.host
    print(f"Connecting to Alice for path sharing at {host}:{path_server_port}...")
    
    connected = False
    for attempt in range(10): # Retry for 10 attempts
        try:
            reader, writer = await asyncio.open_connection(host, path_server_port)
            connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            print(f"Connection attempt {attempt+1}/10 failed: {e}. Retrying in 2s...")
            await asyncio.sleep(2)
            
    if not connected:
        print(f"Failed to connect to Alice after multiple attempts.")
        return

    print(f"Connected to Alice!")

    try:
        for i in range(args.iterations):
            console.print(f"\n[bold magenta]--- Iteration {i+1} ---[/bold magenta]")
            
            # 2. Receive Path
            print(f"Waiting for start signal from Alice...")
            try:
                msg = await recv_msg(reader)
                if msg != "START":
                    print(f"Unexpected message from Alice: {msg}")
                    break
            except asyncio.IncompleteReadError:
                print(f"Alice disconnected.")
                break
                
            print(f"Received Start Signal")
            
            # 3. Run MPC
            with console.status("[yellow]Joining MPC computation...[/yellow]", spinner="dots"):
                # Use local safe function
                result = await safe_join_computation(
                    id=args.party_id,
                    num_parties=args.num_parties,
                    input=bob.grid,
                    port=args.base_port,
                    host=args.host,
                    protocol=protocol
                )
            is_safe = result.is_solved
            status_color = "green" if is_safe else "red"
            status_text = "SAFE" if is_safe else "UNSAFE"
            console.print(f"MPC Result: [bold {status_color}]{status_text}[/bold {status_color}]")
            
            # 4. Receive Info Gain
            console.print("Waiting for Info Gain from Alice...")
            info_gain = await recv_msg(reader)
            total_info_gain += info_gain
            
            console.print(f"Iteration Info Gain: [cyan]{info_gain:.4f}[/cyan]")
            remaining = args.info_budget - total_info_gain
            console.print(f"Total Info Gain: [bold blue]{total_info_gain:.4f}[/bold blue] / {args.info_budget} (Remaining: [bold yellow]{remaining:.4f}[/bold yellow])")
            
            # 5. Sync Barrier
            # console.print("Sending ACK to Alice...", style="dim")
            await send_msg(writer, "ACK")
            
            if total_info_gain > args.info_budget:
                console.print("[bold red]WARNING: Information Gain Budget Exceeded![/bold red]")
            
            # No belief update for Bob anymore
            # print(f"=== Bob {args.party_id} - Updated Belief Map ===")
            # print(str(bob.bayes_map))
            
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        print(f"Closing connection...")
        writer.close()
        await writer.wait_closed()
        print(f"Bob finished.")

# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Bayesian Path Finding")
    parser.add_argument("--role", choices=["alice", "bob"], required=True, help="Role to play")
    parser.add_argument("--party_id", type=int, required=True, help="MPC Player ID (Alice=0, Bob=1..N)")
    parser.add_argument("--num_parties", type=int, required=True, help="Total number of parties")
    parser.add_argument("--host", required=True, help="Host IP/Hostname of Alice (Coordinator)")
    parser.add_argument("--port", type=int, default=5000, help="Port to use (should be base_port + party_id)")
    
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size (NxN)")
    parser.add_argument("--path_length", type=int, default=3, help="Length of path to query")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--protocol", choices=["shamir", "mascot"], default="mascot", help="MPC Protocol")
    parser.add_argument("--seed", type=int, help="Shared seed for deterministic maze generation")
    parser.add_argument("--info_budget", type=float, default=10.0, help="Information Gain Budget for Bob")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Standard MP-SPDZ: All parties must agree on the same Base Port (e.g. 5000)
    # The system then assigns 5000 to P0, 5001 to P1, etc.
    args.base_port = args.port
    
    if args.role == "alice":
        asyncio.run(run_alice(args))
    else:
        asyncio.run(run_bob(args))

if __name__ == "__main__":
    main()

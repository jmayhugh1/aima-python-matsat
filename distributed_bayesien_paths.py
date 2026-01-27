import argparse
import asyncio
import sys
import os
import pickle
import struct
from typing import List, Tuple

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bayesien_paths import Alice, Bob, BayesMap
from private_path_query_utils import Grid, Protocol, join_computation, Path, compile_private_path_query

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

async def run_alice(args):
    print(f"Starting Alice (Player {args.party_id}) on {args.host}:{args.port}...")
    
    start = (0, 0)
    goal = (args.grid_size - 1, args.grid_size - 1)
    
    alice = Alice(start=start, goal=goal, path_lengths=args.path_length, grid_size=args.grid_size)
    protocol = Protocol(args.protocol)
    
    # 1. Start TCP Server for Path Sharing
    # We use the COMMON args.port + 100 so Bob knows where to connect
    # (args.base_port might differ if it includes party_id)
    path_server_port = args.port + 100
    bob_writers = []
    
    connection_event = asyncio.Event()
    expected_bobs = args.num_parties - 1 # Parties - Alice
    
    async def handle_bob(reader, writer):
        addr = writer.get_extra_info('peername')
        print(f"Bob connected from {addr}")
        bob_writers.append(writer)
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

        print("Initial Belief Map:")
        print(str(alice.bayes_map))

        for i in range(args.iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # 1. Plan path
            path, _ = alice.bayes_map.find_highest_entropy_path(alice.path_length)
            print(f"Queried Path: {path.pretty_str()}")
            
            # 2. Share path with Bobs
            print("Broadcasting path to Bobs...")
            broadcast_tasks = [send_msg(w, path) for w in bob_writers]
            await asyncio.gather(*broadcast_tasks)
            
            # 3. Run MPC
            print("Running MPC computation...")
            is_safe = await join_computation(
                id=args.party_id,
                num_parties=args.num_parties,
                input=path,
                port=args.base_port,
                host=args.host,
                protocol=protocol
            )
            print(f"MPC Result: {'SAFE' if is_safe else 'UNSAFE'}")
            
            # 4. Update Belief
            alice.bayes_map.update_probabilities(path, is_safe)
            
            # 5. Visualize
            print("Updated Belief Map:")
            print(str(alice.bayes_map))
            
            if alice.bayes_map.check_viable_path(alice.start, alice.goal):
                print("A viable path exists!")
                # Theoretically send 'Stop' signal or proceed, 
                # but for this loop we'll just finish iterations or exit.
                # For cleaner exit we should probably tell Bobs to stop, but for now we run fixed iterations.
                # Let's just break locally and maybe Bobs handle connection close or we just run all iters.
                # To keep it simple: run all iterations.
    
    print("Alice finished.")

# ==============================================================================
# BOB LOGIC
# ==============================================================================

import random

def generate_partitioned_maze(grid_size: int, seed: int, party_id: int, num_bobs: int) -> Grid:
    """
    Generates a deterministic global maze using 'seed', then returns
    the subset of walls assigned to 'party_id'.
    """
    # 1. Deterministic Generation
    rng = random.Random(seed)
    
    # Simple Randomized Prim's or DFS for Maze Generation
    # Initialize full grid with Walls (1)
    # We'll treat the grid as a graph where nodes are (r, c)
    # To make a proper maze, we often use odd coordinates for cells and even for walls (or similar),
    # but for simplicity on a dense grid:
    # Let's generate a Spannng Tree on the grid cells.
    # Start with all 1s (Obstacles). Carve 0s (Paths).
    
    # Note: 'Grid' class uses 0 for safe, 1 for hazard.
    # For a maze, we want 1s as walls.
    
    map_data = [[1] * grid_size for _ in range(grid_size)]
    
    # Helper to check bounds
    def in_bounds(r, c):
        return 0 <= r < grid_size and 0 <= c < grid_size

    # Carve start and goal immediately
    start = (0, 0)
    map_data[0][0] = 0
    
    # Frontier for Prim's: (r, c)
    frontier = []
    
    def add_neighbors(r, c):
        for dr, dc in [(2,0), (-2,0), (0,2), (0,-2)]:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and map_data[nr][nc] == 1:
                frontier.append((nr, nc, r, c)) 
                
    add_neighbors(0, 0)
    
    # Use RNG for deterministic frontier selection
    while frontier:
        # Pick random edge from frontier
        idx = rng.randint(0, len(frontier) - 1)
        r, c, pr, pc = frontier.pop(idx)
        
        if map_data[r][c] == 0:
            continue
            
        # Carve current cell
        map_data[r][c] = 0
        # Carve wall between current and parent
        wr, wc = (r + pr) // 2, (c + pc) // 2
        map_data[wr][wc] = 0
        
        add_neighbors(r, c)
        
    # Ensure goal is open (it might be wall if simple generation didn't hit it nicely)
    # With odd dimension logic 0,0 to N-1,N-1 usually works if N is odd.
    # If N is even, might need specific carving.
    # We simply force carve the path to goal from nearest 0 if needed or just carve goal.
    map_data[grid_size-1][grid_size-1] = 0
    if grid_size > 1:
        # Simplistic fix to ensure goal connectivity if it was a wall
        # Check neighbors, if all 1, carve one.
        pass # The Maze alg mainly produces a spanning tree, so connectivity is high.

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

async def run_bob(args):
    print(f"Starting Bob (Player {args.party_id})...")
    
    # Generate/Setup Grid
    grid = get_bob_grid(args)
    bob = Bob(grid=grid)
    protocol = Protocol(args.protocol)
    
    print("Bob's Map (Hazards=1):")
    print(grid)
    
    # 1. Connect to Alice for Path Sharing
    # Use common args.port + 100
    path_server_port = args.port + 100
    host = args.host
    print(f"Connecting to Alice at {host}:{path_server_port}...")
    
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
        print("Failed to connect to Alice after multiple attempts.")
        return

    print("Connected to Alice!")

    try:
        for i in range(args.iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # 2. Receive Path
            print("Waiting for path from Alice...")
            try:
                path = await recv_msg(reader)
            except asyncio.IncompleteReadError:
                print("Alice disconnected.")
                break
                
            print(f"Received Path: {path.pretty_str()}")
            
            # 3. Run MPC
            print("Joining MPC computation...")
            is_safe = await join_computation(
                id=args.party_id,
                num_parties=args.num_parties,
                input=bob.grid,
                port=args.base_port,
                host=args.host,
                protocol=protocol
            )
            print(f"MPC Result: {'SAFE' if is_safe else 'UNSAFE'}")
            
            # 4. Update Belief (Now possible!)
            print("Updating belief map...")
            bob.bayes_map.update_probabilities(path, is_safe)
            print("Bob's Updated Belief Map:")
            print(str(bob.bayes_map))
            
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        print("Closing connection...")
        writer.close()
        await writer.wait_closed()
        print("Bob finished.")

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

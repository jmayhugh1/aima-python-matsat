import argparse
import asyncio
import sys
import os
import pickle
import struct
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from private_path_query_utils import (
    Graph,
    Vertex,
    Edge,
    EdgeState,
    PrivatePathInfo,
    Protocol,
    join_computation_find_safe_path,
    join_computation,
    ProgramName,
    Path,
    GraphPath,
    spdz_root
)
from private_path_query_logic import directed_pairs_from_edges, ordered_symbols
from tests.private_path_query_test_helpers import (
    _unknown_domain_edges_from_graph,
    _print_assignments_from_u_vector
)
from rich.console import Console

console = Console(force_terminal=True)

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


from distributed.prm_generator import generate_prm, partition_edges, assign_bob_edge_weights

# ==============================================================================
# GRAPH SETUP
# ==============================================================================

def build_bob_graph(b_id: int, num_parties: int, base_graph: Graph, seed: int) -> Tuple[Graph, dict]:
    """
    Returns the graph partition and exact edge weights for the given Bob.
    Edges chosen for 'soft blocks' receive EdgeState.BLOCKED (generating ~Allowed)
    with custom penalty weights. Other owned edges are TRAVERSABLE.
    """
    partitions = partition_edges(base_graph, num_parties, seed=seed)
    
    if b_id not in partitions:
        return base_graph, {}
        
    bob_owned_edges = partitions[b_id]
    
    # Assign edge penalty weights from a 1.0 budget pool.
    # We alter the seed specifically for each Bob so they block different subsets.
    bob_weights = assign_bob_edge_weights(bob_owned_edges, total_budget=1.0, block_fraction=0.5, seed=seed + b_id)
    
    # Construct the finalized Bob graph mapped onto the MPC
    final_edges = []
    for edge in bob_owned_edges:
        u, v = edge.vertex1.id, edge.vertex2.id
        state = EdgeState.BLOCKED if (u, v) in bob_weights else EdgeState.TRAVERSABLE
        final_edges.append(Edge(edge.vertex1, edge.vertex2, state))
        
    G_bob = Graph(vertices=base_graph.vertices, edges=final_edges)
    return G_bob, bob_weights


# ==============================================================================
# PREDEFINED EXAMPLES
# ==============================================================================

def get_diamond_example(party_id: int, num_parties: int):
    """
    Diamond graph:  0 ---> 1 ---> 3
                    0 ---> 2 ---> 3
    
    Alice wants: 0 -> 3
    Bob 1 (party 1) owns top branch: edges 0<->1, 1<->3  (blocks all, budget=1.0)
    Bob 2 (party 2) owns bottom branch: edges 0<->2, 2<->3 (blocks all, budget=1.0)
    
    Since both branches are blocked, the MaxSAT solver must pick the least-cost path.
    """
    V = 4
    T = 3  # need 2 hops: 0->x->3, so T=3 timesteps
    verts = [Vertex(i) for i in range(V)]
    
    # Full undirected graph (all edges known publicly as UNKNOWN domain)
    all_edges = [
        Edge(verts[0], verts[1], EdgeState.TRAVERSABLE),
        Edge(verts[1], verts[0], EdgeState.TRAVERSABLE),
        Edge(verts[0], verts[2], EdgeState.TRAVERSABLE),
        Edge(verts[2], verts[0], EdgeState.TRAVERSABLE),
        Edge(verts[1], verts[3], EdgeState.TRAVERSABLE),
        Edge(verts[3], verts[1], EdgeState.TRAVERSABLE),
        Edge(verts[2], verts[3], EdgeState.TRAVERSABLE),
        Edge(verts[3], verts[2], EdgeState.TRAVERSABLE),
    ]
    base_graph = Graph(vertices=verts, edges=all_edges)
    start_v = verts[0]
    goal_v = verts[3]
    
    if party_id == 0:
        # Alice: no graph, just start/goal
        return base_graph, start_v, goal_v, V, T, None, None
    elif party_id == 1:
        # Bob 1: owns top branch (0↔1, 1↔3). Budget=1.0
        # Strategy: heavily block forward direction to prevent 0→1→3 path
        bob_edges = [
            Edge(verts[0], verts[1], EdgeState.BLOCKED),
            Edge(verts[1], verts[0], EdgeState.BLOCKED),
            Edge(verts[1], verts[3], EdgeState.BLOCKED),
            Edge(verts[3], verts[1], EdgeState.BLOCKED),
        ]
        bob_graph = Graph(vertices=verts, edges=bob_edges)
        bob_weights = {
            (0, 1): 0.6,   # heavy block on entry
            (1, 3): 0.3,   # moderate block on exit
            (1, 0): 0.05,  # light block on reverse
            (3, 1): 0.05,  # light block on reverse
        }  # sum = 1.0
        return base_graph, start_v, goal_v, V, T, bob_graph, bob_weights
    elif party_id == 2:
        # Bob 2: owns bottom branch (0↔2, 2↔3). Budget=1.0
        # Strategy: concentrate on blocking the 2→3 chokepoint
        bob_edges = [
            Edge(verts[0], verts[2], EdgeState.BLOCKED),
            Edge(verts[2], verts[0], EdgeState.BLOCKED),
            Edge(verts[2], verts[3], EdgeState.BLOCKED),
            Edge(verts[3], verts[2], EdgeState.BLOCKED),
        ]
        bob_graph = Graph(vertices=verts, edges=bob_edges)
        bob_weights = {
            (0, 2): 0.1,   # light block on entry
            (2, 3): 0.7,   # heavy block on chokepoint
            (2, 0): 0.1,   # light block on reverse
            (3, 2): 0.1,   # light block on reverse
        }  # sum = 1.0
        return base_graph, start_v, goal_v, V, T, bob_graph, bob_weights
    else:
        raise ValueError(f"Diamond example only supports 3 parties, got party_id={party_id}")

# ==============================================================================
# RUNNERS
# ==============================================================================

async def run_sync_server(port, expected_clients):
    """Starts the TCP server and waits for clients to connect, then broadcasts START."""
    writers = []
    readers = []
    connection_event = asyncio.Event()
    start_event = asyncio.Event()

    async def handle_client(reader, writer):
        addr = writer.get_extra_info('peername')
        print(f"Client connected from {addr}")
        writers.append(writer)
        readers.append(reader)
        if len(writers) == expected_clients:
            connection_event.set()
        await start_event.wait()

    server = await asyncio.start_server(handle_client, '0.0.0.0', port)
    console.print(f"[bold cyan]Server listening for {expected_clients} clients on port {port}...[/bold cyan]")
    
    await connection_event.wait()
    console.print("[bold green]All clients connected! Broadcasting START...[/bold green]")
    
    broadcast_tasks = [send_msg(w, "START") for w in writers]
    await asyncio.gather(*broadcast_tasks)

    start_event.set()
    await asyncio.sleep(1)
    
    for w in writers:
        try:
            w.close()
        except Exception:
            pass
    server.close()



async def wait_for_start(host, port):
    """Connects to the orchestrator (Bob 1) and waits for the START signal."""
    connected = False
    for attempt in range(10):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            print(f"Connection attempt {attempt+1}/10 failed: {e}. Retrying in 2s...")
            await asyncio.sleep(2)
            
    if not connected:
        raise RuntimeError(f"Failed to connect to Bob 1 after multiple attempts.")

    console.print("[bold yellow]Waiting for START signal...[/bold yellow]")
    msg = await recv_msg(reader)
    if msg != "START":
        raise RuntimeError(f"Unexpected message: {msg}")
    
    console.print("[bold green]Received START Signal[/bold green]")
    writer.close()
    await writer.wait_closed()


async def execute_query(args):
    """Executes the actual MP-SPDZ protocol based on mode and role."""
    
    # Resolve graph setup: predefined example or PRM generator
    if args.example == "diamond":
        base_graph, start_v, goal_v, V, T, bob_graph, bob_weights = get_diamond_example(
            args.party_id, args.num_parties
        )
        console.print(f"[bold yellow]Using predefined DIAMOND graph: 0→1→3, 0→2→3[/bold yellow]")
    else:
        V = args.num_vertices
        T = args.horizon
        console.print(f"[dim]Building PRM with V={V}, radius={args.prm_radius}, seed={args.seed}[/dim]")
        base_graph = generate_prm(V, radius=args.prm_radius, seed=args.seed)
        start_v = Vertex(0)
        goal_v = Vertex(V - 1)
        bob_graph = None
        bob_weights = None
    
    public_domain_edges = _unknown_domain_edges_from_graph(base_graph)
    
    # Calculate row sizes (needed by compile_find_safe_path)
    from private_path_query_logic import build_bob_q, build_physics_q, build_alice_q
    compact_pairs = directed_pairs_from_edges(public_domain_edges, V)
    syms = ordered_symbols(T, V, directed_pairs=compact_pairs)
    
    num_bobs = args.num_parties - 1
    
    # Dummy row calculations
    q_physics, _ = build_physics_q(T, V, symbols=syms, directed_pairs=compact_pairs)
    q_alice, _ = build_alice_q(start_v.id, goal_v.id, T, V, symbols=syms)
    alice_rows = q_physics.shape[0] + q_alice.shape[0]
    
    # Build rows_per_id: [alice_rows, bob1_rows, bob2_rows, ...]
    bob_rows_list = []
    for b_id in range(1, args.num_parties):
        if args.example == "diamond":
            _, _, _, _, _, bg, _ = get_diamond_example(b_id, args.num_parties)
        else:
            bg, _ = build_bob_graph(b_id, args.num_parties, base_graph, args.seed)
        q_b, _ = build_bob_q(bg.to_directed_edges(), T, V, symbols=syms, directed_pairs=compact_pairs)
        bob_rows_list.append(q_b.shape[0])
        
    rows_per_id = [alice_rows] + bob_rows_list
    
    info = PrivatePathInfo(
        num_parties=args.num_parties,
        T=T,
        V=V,
        rows_per_id=rows_per_id,
        use_weight_vector=True,
        use_edge_domain=True,
        edge_domain_edges=public_domain_edges
    )

    protocol = Protocol(args.protocol)
    result = None

    if args.mode == "matsat":
        if args.party_id == 0:
            console.print(f"[bold cyan]Alice (party 0) inputs: start={start_v.id}, goal={goal_v.id}[/bold cyan]")
            result = await join_computation_find_safe_path(
                id=args.party_id,
                private_path_info=info,
                start=start_v,
                goal=goal_v,
                port=args.base_port,
                host=args.host,
                protocol=protocol,
                compile_program=True,
                weighted=True,
            )
        else:
            # For predefined examples, use the pre-built graph; for PRM, build it
            if args.example:
                b_graph, b_weights = bob_graph, bob_weights
            else:
                b_graph, b_weights = build_bob_graph(args.party_id, args.num_parties, base_graph, args.seed)
            
            console.print(f"[dim]Bob {args.party_id} inputs graph edges: {[str(e) for e in b_graph.edges]}[/dim]")
            console.print(f"[dim]Bob {args.party_id} edge weight penalties: {b_weights}[/dim]")
            
            result = await join_computation_find_safe_path(
                id=args.party_id,
                private_path_info=info,
                graph=b_graph,
                port=args.base_port,
                host=args.host,
                protocol=protocol,
                compile_program=False,
                weighted=True,
            )

        # Print MatSat Results
        status_color = "green" if result and result.is_solved else "red"
        status_text = "SOLVED (Safe)" if result and result.is_solved else "UNSOLVED"
        console.print(f"\n[bold {status_color}]Result: {status_text}[/bold {status_color}]")
        
        if result and result.u_vector and args.party_id == 0:
            console.print("\n[bold cyan]Solver Assignment U-Vector Decode:[/bold cyan]")
            _print_assignments_from_u_vector(
                u_vector=result.u_vector,
                T=T,
                V=V,
                symbols=syms
            )

    elif args.mode == "verifier":
        # Note: verifier mode requires recompiling verifier program.
        # But this script uses join_computation directly.
        from private_path_query_utils import compile_verifier
        if args.party_id == 0:
            # Recompile verifier if Alice
            await compile_verifier(args.num_parties, V, T, is_graph=True)
            
            # Alice wants to verify this strict path
            test_path = GraphPath(start=start_v, moves=[
                Edge(start_v, Vertex(1)),
                Edge(Vertex(1), Vertex(2)),
                Edge(Vertex(2), goal_v)
            ])
            # Verifier graph mode expects `query_size` number of edges from Alice: each a (u, v) pair
            path_str_payload = f"{start_v.id}\n1\n1\n2\n2\n3\n"
            console.print(f"[dim]Alice inputs test path payload: {path_str_payload}[/dim]")
            result = await join_computation(
                id=args.party_id,
                num_parties=args.num_parties,
                input=path_str_payload,
                port=args.base_port,
                host=args.host,
                protocol=protocol,
                program_name=ProgramName.VERIFIER
            )
        else:
            bob_graph, _ = build_bob_graph(args.party_id, args.num_parties, base_graph, args.seed)
            console.print(f"[dim]Bob {args.party_id} inputs PRM partition graph for Verification. Edges: {[str(e) for e in bob_graph.edges]}[/dim]")
            
            # For the verifier, Bob inputs a 1D array representing the flattened V*V adjacency matrix
            # where 2 = traversable, 1 = blocked, 0 = no edge.
            adj_list = bob_graph.adjacency_list
            bob_payload_str = "\n".join(str(int(adj_list[i][j])) for i in range(V) for j in range(V)) + "\n"
            
            result = await join_computation(
                id=args.party_id,
                num_parties=args.num_parties,
                input=bob_payload_str,
                port=args.base_port,
                host=args.host,
                protocol=protocol,
                program_name=ProgramName.VERIFIER
            )
            
        status_color = "green" if result and result.is_solved else "red"
        status_text = "SAFE" if result and result.is_solved else "UNSAFE / BLOCKED"
        console.print(f"\n[bold {status_color}]Verifier Result: {status_text}[/bold {status_color}]")


async def main_async(args):
    console.rule(f"[bold magenta]Starting {args.role.capitalize()} (Party {args.party_id}) - Mode: {args.mode}[/bold magenta]")
    
    sync_port = args.port + 100
    
    # 1. Orchestration — Alice (party 0) is the orchestrator
    if args.party_id == 0:
        # Alice starts the server and waits for everyone else
        server_task = asyncio.create_task(run_sync_server(sync_port, args.num_parties - 1))
        await asyncio.sleep(0.5)
        console.print("[dim]Alice (party 0) orchestrator ready. Waiting for Bobs...[/dim]")
        await server_task
    else:
        # Bobs connect to Alice's orchestrator
        await wait_for_start(args.host, sync_port)

    # 2. Execution
    await execute_query(args)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed MatSat/Verifier Run")
    parser.add_argument("--role", choices=["alice", "bob"], required=True, help="Role to play")
    parser.add_argument("--party_id", type=int, required=True, help="MPC Player ID (Alice=0, Bob=1..N)")
    parser.add_argument("--num_parties", type=int, required=True, help="Total number of parties")
    parser.add_argument("--host", required=True, help="Host IP/Hostname of Alice (Coordinator)")
    parser.add_argument("--port", type=int, default=5000, help="Base Port to use (MP-SPDZ default 5000)")
    parser.add_argument("--protocol", choices=["shamir", "mascot"], default="mascot", help="MPC Protocol")
    parser.add_argument("--mode", choices=["matsat", "verifier"], default="matsat", help="Action to perform")
    parser.add_argument("--example", choices=["diamond"], default=None, help="Use a predefined graph example")
    
    # Graph Generator arguments (used when --example is not set)
    parser.add_argument("--num_vertices", type=int, default=6, help="Vertices in the PRM graph")
    parser.add_argument("--prm_radius", type=float, default=0.5, help="Graph generation radius connection threshold")
    parser.add_argument("--seed", type=int, default=42, help="Seed for identical random PRM graph creation")
    parser.add_argument("--horizon", type=int, default=5, help="Time Horizon (T) allowable path length")
    
    args = parser.parse_args()
    args.base_port = args.port
    return args

if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))

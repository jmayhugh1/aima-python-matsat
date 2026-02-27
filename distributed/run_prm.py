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
    GraphPath,
    spdz_root,
    compile_verifier,
    compile_find_safe_path
)
from private_path_query_logic import directed_pairs_from_edges, ordered_symbols, build_physics_q, build_alice_q, build_bob_q, assignment_from_u_vector, decode_path_from_assignment
from tests.private_path_query_test_helpers import (
    _unknown_domain_edges_from_graph,
    _print_assignments_from_u_vector
)
from tests.prm_graph_configs import (
    GRAPH_CONFIGS,
    BLOCKING_PERCENTAGES,
    create_edges_with_random_safe_path,
)
from bayesien_paths import BayesGraphMap
from rich.console import Console
from typing import List, Optional
import json
from distributed.graph_embedder import generate_spatial_embedding, get_fountain_collisions, visualize_embedding

console = Console(force_terminal=True)

def _create_alice_graph(vertices, edge_pairs) -> Graph:
    """Create Alice's belief graph with all edges as TRAVERSABLE."""
    edges = [Edge(vertices[v1], vertices[v2], EdgeState.TRAVERSABLE) for v1, v2 in edge_pairs]
    return Graph(vertices=vertices, edges=edges)

def _path_to_str(start_id: int, path: Optional[GraphPath]) -> str:
    """Convert a GraphPath to a string representation."""
    if path is None:
        return "NO_PATH"
    return " -> ".join([str(start_id)] + [str(e.vertex2.id) for e in path.moves])

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
# RUNNERS
# ==============================================================================

async def run_sync_server(port, expected_clients, broadcast_msg="START"):
    """Starts the TCP server and waits for clients to connect, then broadcasts message."""
    writers = []
    readers = []
    connection_event = asyncio.Event()
    start_event = asyncio.Event()

    async def handle_client(reader, writer):
        addr = writer.get_extra_info('peername')
        console.print(f"[dim]Client connected from {addr}[/dim]")
        writers.append(writer)
        readers.append(reader)
        if len(writers) == expected_clients:
            connection_event.set()
        await start_event.wait()

    server = await asyncio.start_server(handle_client, '0.0.0.0', port)
    console.print(f"[bold cyan]Server listening for {expected_clients} clients on port {port}...[/bold cyan]")
    
    await connection_event.wait()
    
    broadcast_tasks = [send_msg(w, broadcast_msg) for w in writers]
    await asyncio.gather(*broadcast_tasks)

    start_event.set()
    await asyncio.sleep(0.5)
    
    for w in writers:
        try:
            w.close()
            await w.wait_closed()
        except Exception:
            pass
    server.close()
    await server.wait_closed()

async def wait_for_signal(host, port):
    """Connects to the orchestrator and waits for a signal."""
    connected = False
    for attempt in range(60):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            console.print(f"[dim]Connection attempt {attempt+1}/60 failed. Retrying in 5s...[/dim]")
            await asyncio.sleep(5)
            
    if not connected:
        raise RuntimeError(f"Failed to connect to orchestrator.")

    msg = await recv_msg(reader)
    writer.close()
    await writer.wait_closed()
    return msg


async def execute_prm_query(args):
    """Executes the PRM benchmark protocol (iterative verifier or matsat)."""
    
    if args.graph_name not in GRAPH_CONFIGS:
        console.print(f"[bold red]Unknown graph name: {args.graph_name}. Available: {list(GRAPH_CONFIGS.keys())}[/bold red]")
        sys.exit(1)
        
    config = GRAPH_CONFIGS[args.graph_name]
    num_nodes = config["num_nodes"]
    edge_pairs = config["edge_pairs"]
    start_id = config["start_id"]
    goal_id = config["goal_id"]
    max_path_length = config["max_path_length"]
    
    vertices = [Vertex(id=i) for i in range(num_nodes)]
    start_v = vertices[start_id]
    goal_v = vertices[goal_id]

    protocol = Protocol(args.protocol)
    
    # 1. Determine safe path BEFORE spatial embedding for path-aware placement
    from tests.prm_graph_configs import find_random_path
    
    # We use a temporary graph to find a path that doesn't cross fountains
    # But wait, we don't have coordinates yet! 
    # The constraint is: find ANY topological path, then FORCE the embedder to make it safe.
    
    safe_path_nodes, safe_path_edge_indices = find_random_path(
        num_nodes=num_nodes,
        edge_pairs=edge_pairs,
        start_id=start_id,
        goal_id=goal_id,
        max_length=max_path_length,
        seed=args.seed
    )

    # 2. Handle spatial embedding if requested
    coords_map = None
    if args.spatial_bounds:
        console.print(f"[bold cyan]Generating path-aware spatial embedding (seed={args.seed})...[/bold cyan]")
        try:
            temp_graph = Graph(vertices=vertices, edges=[Edge(vertices[v1], vertices[v2], EdgeState.TRAVERSABLE) for v1, v2 in edge_pairs])
            
            # Pin start and goal nodes to specific regions as requested
            node_regions = {str(start_id): "top_left", str(goal_id): "bottom_right"}
            
            # Pass safe_path_edge_indices to ensure the ground-truth path is fountain-free
            coords_map = generate_spatial_embedding(
                temp_graph, 
                args.spatial_bounds, 
                strict=False, 
                seed=args.seed, 
                min_dist_m=args.min_dist, 
                node_regions=node_regions,
                safe_path_edge_indices=safe_path_edge_indices
            )
            
            # Save coordinates
            example_dir = os.path.join(current_dir, "examples", args.graph_name)
            os.makedirs(example_dir, exist_ok=True)
            coords_path = os.path.join(example_dir, "coordinates.json")
            with open(coords_path, 'w') as f:
                json.dump(coords_map, f, indent=2)
            
            # Identify which edges intersect with fountains (any edge, including non-path edges)
            mandatory_blocked_indices = set(get_fountain_collisions(temp_graph, coords_map, args.spatial_bounds))
            if mandatory_blocked_indices:
                # The safe path edges should NOT be in this set due to our embedder constraints
                console.print(f"[bold yellow]Identified {len(mandatory_blocked_indices)} edges intersecting with fountains as mandatory obstacles.[/bold yellow]")
            
        except Exception as e:
            console.print(f"[bold red]Failed to generate or save spatial embedding: {e}[/bold red]")
            sys.exit(1)

    # 3. Generate ground truth graph using the pre-selected safe path
    ground_truth_edges = []
    # Similar to create_edges_with_random_safe_path but using our pre-selected data
    import random
    random.seed(args.seed + 1000)
    
    non_path_indices = [i for i in range(len(edge_pairs)) if i not in safe_path_edge_indices]
    # Any edge crossing a fountain MUST be blocked
    if coords_map:
        mandatory_blocked_indices = set(get_fountain_collisions(temp_graph, coords_map, args.spatial_bounds))
        non_path_indices = [i for i in non_path_indices if i not in mandatory_blocked_indices]

    num_to_block = int(len(non_path_indices) * args.block_percent / 100)
    indices_to_block = set(random.sample(non_path_indices, num_to_block))
    
    if coords_map:
        indices_to_block.update(mandatory_blocked_indices)

    for i, (v1, v2) in enumerate(edge_pairs):
        if i in safe_path_edge_indices:
            state = EdgeState.TRAVERSABLE
        elif i in indices_to_block:
            state = EdgeState.BLOCKED
        else:
            state = EdgeState.TRAVERSABLE
        ground_truth_edges.append(Edge(vertices[v1], vertices[v2], state))
    
    bob_graph = Graph(vertices=vertices, edges=ground_truth_edges)
    
    # 3. Initial visualization showing blocked vs safe edges
    if args.spatial_bounds and args.visualize:
        example_dir = os.path.join(current_dir, "examples", args.graph_name)
        viz_path = os.path.join(example_dir, "embedding_initial.png")
        # Alice visualizes the ground truth for debugging/benchmarking
        visualize_embedding(bob_graph, coords_map, args.spatial_bounds, viz_path)
    alice_graph = _create_alice_graph(vertices, edge_pairs)

    console.print(f"\n[bold yellow]Graph: {args.graph_name} ({num_nodes}v, {len(edge_pairs)}e)[/bold yellow]")
    console.print(f"[bold yellow]Blocking: {args.block_percent}% | Seed: {args.seed}[/bold yellow]")
    
    if args.party_id == 0:
        console.print(f"[bold green]True random safe path: {' -> '.join(str(n) for n in safe_path_nodes)}[/bold green]\n")

    if args.mode == "iterative_verifier":
        await execute_iterative_verifier(args, alice_graph, bob_graph, start_v, goal_v, max_path_length, protocol, coords_map)
    elif args.mode == "matsat_solver":
        await execute_matsat_solver(args, alice_graph, bob_graph, start_v, goal_v, protocol, coords_map)


async def execute_iterative_verifier(args, alice_graph, bob_graph, start_v, goal_v, max_path_length, protocol, coords_map=None):
    """Runs the verifier repeatedly until a safe path is found (or max iterations)."""
    
    sync_port = args.port + 100
    num_bobs = args.num_parties - 1
    
    if args.party_id == 0:
        # Initialize Bayesian tracker
        alice_belief = BayesGraphMap(alice_graph, p_init=0.3)
        strategies = {
            "entropy": lambda bm, s, g, ml: bm.find_highest_entropy_path(s, g, ml),
            "likelihood": lambda bm, s, g, ml: bm.find_highes_likelihood_safe_path(s, g, ml),
            "random": lambda bm, s, g, ml: bm.find_random_path(s, g, ml),
        }
        
        if args.strategy not in strategies:
            console.print(f"[bold red]Unknown strategy: {args.strategy}[/bold red]")
            sys.exit(1)
            
        path_finder = strategies[args.strategy]
        
        total_info_gain = 0.0
        found_safe = False
        
        for iteration in range(args.max_iterations):
            console.rule(f"[bold magenta]Iteration {iteration + 1}[/bold magenta]")
            
            # 1. Find a path using the local strategy
            _, path = path_finder(alice_belief, start_v, goal_v, max_path_length)
            
            if path is None:
                console.print("[bold red]Alice could not find any path! Stopping.[/bold red]")
                server_task = asyncio.create_task(run_sync_server(sync_port, num_bobs, "STOP"))
                await server_task
                break
                
            path_str = _path_to_str(start_v.id, path)
            console.print(f"[bold cyan]Alice proposing path:[/bold cyan] {path_str}")
            
            # 2. Compile verifier dynamically for this specific path length (matching test_run behavior)
            query_size = len(path.moves)
            console.print(f"[dim]Compiling verifier for query_size={query_size}...[/dim]")
            await compile_verifier(args.num_parties, len(alice_graph.vertices), query_size, is_graph=True, iteration_no=iteration)
            
            # 3. Tell Bobs to continue and run Verifier
            server_task = asyncio.create_task(run_sync_server(sync_port, num_bobs, "CONTINUE"))
            await server_task
            
            # Encode path as payload without padding since we specifically compiled for this size
            path_str_payload = str(path) + "\n"
            
            # 4. Run MP-SPDZ Verifier
            result = await join_computation(
                id=0,
                num_parties=args.num_parties,
                input=path_str_payload,
                port=args.base_port,
                host=args.host,
                protocol=protocol,
                program_name=ProgramName.VERIFIER
            )
            
            # 5. Process results
            is_safe = result.is_solved
            info_gain = result.information_gain
            total_info_gain += info_gain
            
            status_color = "green" if is_safe else "red"
            console.print(f"Result: [bold {status_color}]{'SAFE' if is_safe else 'UNSAFE/BLOCKED'}[/bold {status_color}] | Info Gain: {info_gain:.4f}")
            
            alice_belief.update_probabilities(path, safe=is_safe)
            
            if alice_belief.check_viable_path(start_v, goal_v):
                console.print(f"\n[bold green]✓ SUCCESS: Viable path found in {iteration + 1} iterations! Total info gain: {total_info_gain:.4f}[/bold green]")
                # Tell Bobs we're done
                server_task = asyncio.create_task(run_sync_server(sync_port + 1, num_bobs, "STOP"))
                await server_task
                found_safe = True
                
                # Save physical path if spatial bounds are present
                if args.spatial_bounds:
                    example_dir = os.path.join(current_dir, "examples", args.graph_name)
                    # We can't easily get the 'path' variable here in the right format for waypoints
                    # But we can reconstruct it or wait until the next iteration.
                    # Actually, we have 'path' from the loop.
                    node_ids = [start_v.id] + [e.vertex2.id for e in path.moves]
                    waypoints = [[coords_map[str(nid)]["lat"], coords_map[str(nid)]["lon"]] for nid in node_ids]
                    wp_path = os.path.join(example_dir, "waypoints.json")
                    with open(wp_path, 'w') as f:
                        json.dump(waypoints, f, indent=2)
                    console.print(f"[bold green]Saved physical waypoints to {wp_path}[/bold green]")
                    
                    if args.visualize:
                        viz_path = os.path.join(example_dir, "embedding_solved.png")
                        # Use bob_graph for ground-truth visualization colors
                        visualize_embedding(bob_graph, coords_map, args.spatial_bounds, viz_path, path=node_ids)
                break
                
            # Send a sync to clear Bobs for the next loop (prevent out-of-sync)
            server_task = asyncio.create_task(run_sync_server(sync_port + 1, num_bobs, "SYNC"))
            await server_task

        if not found_safe:
            console.print(f"\n[bold red]✗ FAILED: Reached max iterations ({args.max_iterations}) without finding safe path.[/bold red]")
            # Ensure Bobs stop
            server_task = asyncio.create_task(run_sync_server(sync_port + 1, num_bobs, "STOP"))
            await server_task

    else:
        # Bob Iterator Loop
        V = len(bob_graph.vertices)
        
        while True:
            # Wait for Alice's instruction (CONTINUE or STOP)
            msg = await wait_for_signal(args.host, sync_port)
            if msg == "STOP":
                console.print("[bold green]Received STOP signal from Alice. Exiting.[/bold green]")
                break
            elif msg == "CONTINUE":
                console.print("\n[dim]Alice initiated a new verifier query. Running MPC...[/dim]")
                
                # Input PRM block graph
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
                console.print(f"Result: [bold {status_color}]{'SAFE' if result and result.is_solved else 'UNSAFE/BLOCKED'}[/bold {status_color}]")
                
                # Wait for Alice to process and either loop again or stop
                sync_msg = await wait_for_signal(args.host, sync_port + 1)
                if sync_msg == "STOP":
                    console.print("[bold green]Received STOP signal after iteration. Exiting.[/bold green]")
                    break


async def execute_matsat_solver(args, alice_graph, bob_graph, start_v, goal_v, protocol, coords_map=None):
    """Runs the MatSAT solver in one hit on the blocked PRM graph."""
    
    # 1. Setup PRM graph variables
    T = alice_graph._max_path_length_cache if hasattr(alice_graph, '_max_path_length_cache') else GRAPH_CONFIGS[args.graph_name]["max_path_length"]
    V = len(alice_graph.vertices)
    
    from private_path_query_logic import num_bits
    
    public_domain_edges = _unknown_domain_edges_from_graph(alice_graph)
    compact_pairs = directed_pairs_from_edges(public_domain_edges, V)
    syms = ordered_symbols(T, V, directed_pairs=compact_pairs)
    
    # Calculate rows (same as in ordinary run.py)
    # We must mock PrivatePathInfo or just use `private_path_query_logic.physics_bits` directly to count rows.
    # Actually, the logic in `run.py` does:
    q_physics, _ = build_physics_q(T, V, symbols=syms, directed_pairs=compact_pairs)
    
    # `build_alice_q` in `private_path_query_logic.py` takes PrivatePathInfo, but we can't make it unless we know lengths.
    # So we call the internal `_build_alice_q`:
    from private_path_query_logic import _build_alice_q, _build_bob_q
    
    q_alice, _ = _build_alice_q(start_v.id, goal_v.id, T, V, symbols=syms)
    alice_rows = q_physics.shape[0] + q_alice.shape[0]
    
    bob_rows_list = []
    for b_id in range(1, args.num_parties):
        # Mirror undirected edges to directed arcs for the solver
        directed_edges = bob_graph.to_directed_edges()
        q_b, _ = _build_bob_q(directed_edges, T, V, symbols=syms, directed_pairs=compact_pairs)
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
    
    sync_port = args.port + 100

    if args.party_id == 0:
        console.print("[bold cyan]Alice (party 0) compiling MatSAT solver...[/bold cyan]")
        ok = await compile_find_safe_path(private_path_info=info, weighted=True)
        if not ok:
            console.print("[bold red]Alice compilation failed![/bold red]")
            sys.exit(1)
            
        server_task = asyncio.create_task(run_sync_server(sync_port, args.num_parties - 1, "START"))
        await asyncio.sleep(2.0)
        console.print("[dim]Alice (party 0) orchestrator ready. Waiting for Bobs...[/dim]")
        await server_task
        
        console.print(f"[bold cyan]Alice (party 0) executing MatSAT: start={start_v.id}, goal={goal_v.id}[/bold cyan]")
        result = await join_computation_find_safe_path(
            id=0,
            private_path_info=info,
            start=start_v,
            goal=goal_v,
            port=args.base_port,
            host=args.host,
            protocol=protocol,
            compile_program=False,
            weighted=True,
        )
        
        status_color = "green" if result and result.is_solved else "red"
        status_text = "SOLVED (Safe)" if result and result.is_solved else "UNSOLVED"
        console.print(f"\n[bold {status_color}]Result: {status_text}[/bold {status_color}]")
        
        if result and result.u_vector:
            console.print("\n[bold cyan]Solver Assignment U-Vector Decode:[/bold cyan]")
            _print_assignments_from_u_vector(u_vector=result.u_vector, T=T, V=V, symbols=syms)
            assignment = assignment_from_u_vector(u_vector=result.u_vector, T=T, V=V, symbols=syms)
            path = decode_path_from_assignment(assignment, T=T, V=V)
            console.print(f"\n[bold green]✓ EXTRACTED OPTIMAL PATH: {path}[/bold green]")

            # Save physical path if spatial bounds are present
            if args.spatial_bounds:
                example_dir = os.path.join(current_dir, "examples", args.graph_name)
                waypoints = [[coords_map[str(nid)]["lat"], coords_map[str(nid)]["lon"]] for nid in path]
                wp_path = os.path.join(example_dir, "waypoints.json")
                with open(wp_path, 'w') as f:
                    json.dump(waypoints, f, indent=2)
                console.print(f"[bold green]Saved physical waypoints to {wp_path}[/bold green]")
                
                if args.visualize:
                    viz_path = os.path.join(example_dir, "embedding_solved.png")
                    # Use bob_graph for ground-truth visualization colors
                    visualize_embedding(bob_graph, coords_map, args.spatial_bounds, viz_path, path=path)

    else:
        await wait_for_signal(args.host, sync_port)
        
        console.print(f"[dim]Bob {args.party_id} inputs PRM graph edges: {[str(e) for e in bob_graph.edges]}[/dim]")
        
        result = await join_computation_find_safe_path(
            id=args.party_id,
            private_path_info=info,
            graph=bob_graph,
            port=args.base_port,
            host=args.host,
            protocol=protocol,
            compile_program=False,
            weighted=True,
        )
        
        status_color = "green" if result and result.is_solved else "red"
        status_text = "SOLVED (Safe)" if result and result.is_solved else "UNSOLVED"
        console.print(f"\n[bold {status_color}]Result: {status_text}[/bold {status_color}]")


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed MatSat/Verifier PRM Benchmarker")
    parser.add_argument("--role", choices=["alice", "bob"], help="Role to play")
    parser.add_argument("--party_id", type=int, help="MPC Player ID (Alice=0, Bob=1..N)")
    parser.add_argument("--num_parties", type=int, help="Total number of parties")
    parser.add_argument("--host", help="Host IP/Hostname of Alice (Coordinator)")
    parser.add_argument("--port", type=int, default=5000, help="Base Port to use (MP-SPDZ default 5000)")
    parser.add_argument("--protocol", choices=["shamir", "mascot"], default="mascot", help="MPC Protocol")
    
    # PRM benchmark specific
    parser.add_argument("--mode", choices=["iterative_verifier", "matsat_solver"], default="iterative_verifier", help="Benchmark logic to perform")
    parser.add_argument("--graph_name", type=str, default="5v_6e", help="Name of PRM graph from configs (e.g. 5v_6e, 6v_7e)")
    parser.add_argument("--block_percent", type=int, default=25, help="Percentage of optimal path edges blocked (0, 25, 50, 75, 100)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for PRM map layout generation")
    
    # Iterative Verifier specific
    parser.add_argument("--strategy", type=str, choices=["entropy", "likelihood", "random"], default="entropy", help="Strategy for traversing the Bayes belief map")
    parser.add_argument("--max_iterations", type=int, default=20, help="Max iterations before giving up in iterative verifier")
    
    # Spatial / Visualization
    parser.add_argument("--spatial_bounds", type=str, help="Path to spatial_bounds.json")
    parser.add_argument("--visualize", action="store_true", help="Generate satellite visualization of graph and paths")
    parser.add_argument("--min_dist", type=float, default=6.0, help="Minimum distance in meters between graph nodes")
    
    args = parser.parse_args()
    args.base_port = args.port
    return args

if __name__ == "__main__":
    asyncio.run(execute_prm_query(parse_args()))

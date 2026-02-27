import argparse
import asyncio
import sys
import os
import pickle
import struct
import json
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
from private_path_query_logic import assignment_from_u_vector, decode_path_from_assignment
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


from distributed.examples import load_example_from_json

# ==============================================================================
# GRAPH SETUP
# ==============================================================================


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
    """Connects to the orchestrator (Alice / party 0) and waits for the START signal."""
    connected = False
    for attempt in range(60):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            print(f"Connection attempt {attempt+1}/60 failed: {e}. Retrying in 5s (waiting for Alice compilation)...")
            await asyncio.sleep(5)
            
    if not connected:
        raise RuntimeError(f"Failed to connect to Alice orchestrator after multiple attempts.")

    console.print("[bold yellow]Waiting for START signal...[/bold yellow]")
    msg = await recv_msg(reader)
    if msg != "START":
        raise RuntimeError(f"Unexpected message: {msg}")
    
    console.print("[bold green]Received START Signal[/bold green]")
    writer.close()
    await writer.wait_closed()


async def execute_query(args):
    """Executes the actual MP-SPDZ protocol or Waypoint Mission based on mode and role."""
    
    if args.mode == "execute_waypoints":
        # 3. Only the physically connected node (usually Alice, or invoked directly) executes ways
        if args.party_id is not None and args.party_id != 0:
            console.print("[dim]Execution is handled by the primary connected node (party 0). Bobs idle...[/dim]")
            return

        out_path = args.out_path
        if not out_path and args.example:
            if args.example in ["diamond", "sat"]:
                waypoints_in_path = os.path.join(current_dir, "examples", args.example, "waypoints.json")
            else: 
                waypoints_in_path = os.path.join(os.path.dirname(args.example), "waypoints.json")
                
        try:
            with open(waypoints_in_path, 'r') as f:
                physical_waypoints = json.load(f)
            console.print(f"[bold green]Loaded mapped waypoints successfully from {waypoints_in_path}[/bold green]")
        except FileNotFoundError:
            console.print(f"[bold red]Error: No previously computed waypoints.json found at {waypoints_in_path}. Run MatSAT solver first![/bold red]")
            return
            
        try:
            console.print("\n[bold cyan]Connecting to Surveyor ASV to execute waypoints...[/bold cyan]")
            # Append local submodule path if available
            sys.path.append(os.path.join(parent_dir, "searobotics_surveyor"))
            from surveyor_lib.surveyor import Surveyor
            
            host = '192.168.0.50'
            port = 8003
            throttle = 60
            
            # Read ERP from spatial bounds or default
            erp = tuple(physical_waypoints[0])
            if args.spatial_bounds:
                try:
                    with open(args.spatial_bounds, 'r') as f:
                        bounds = json.load(f)
                    erp = tuple(bounds.get("erp", physical_waypoints[0]))
                except Exception as e:
                    console.print(f"[bold yellow]Failed to read ERP from bounds, defaulting to first waypoint: {e}[/bold yellow]")
            
            boat = Surveyor(host=host, port=port, sensors_to_use=[], record=False)
            with boat:
                console.print("[bold green]Surveyor Connected![/bold green]")
                boat.send_waypoints(waypoints=physical_waypoints, erp=erp, throttle=throttle)
                console.print(f"Uploaded {len(physical_waypoints)} waypoints. ERP set to {erp}")
                
                console.print("[bold cyan]Starting Mission on Robot...[/bold cyan]")
                boat.set_waypoint_mode()
                
                import time
                while True:
                    state = boat.get_state()
                    mode = state.get('Control Mode', 'Unknown')
                    if mode == "Standby":
                        console.print("\n[bold green]Waypoint mission completed (Returned to Standby)![/bold green]")
                        break
                    time.sleep(2)
                    
        except Exception as e:
            console.print(f"[bold red]Surveyor execution failed: {e}[/bold red]")
            
        return
        
    # Resolve graph setup: predefined example required for MPC nodes
    if not args.example:
        console.print("[bold red]An --example JSON configuration is strictly required now. Exiting.[/bold red]")
        sys.exit(1)
        
    if args.example in ["diamond", "sat"]:
        # Backwards compatibility for the test scripts
        json_path = os.path.join(current_dir, "examples", f"{args.example}", "graph.json")
    else:
        json_path = args.example
        
    base_graph, start_v, goal_v, V, T, bob_graph, bob_weights = load_example_from_json(
        json_path, args.party_id, args.num_parties
    )
    console.print(f"[bold yellow]Using JSON graph from {json_path}[/bold yellow]")
    
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
        if not args.example:
            console.print("[bold red]An --example JSON configuration is strictly required. Exiting.[/bold red]")
            sys.exit(1)
            
        if args.example in ["diamond", "sat"]:
            json_path = os.path.join(current_dir, "examples", f"{args.example}", "graph.json")
        else:
            json_path = args.example
        _, _, _, _, _, bg, _ = load_example_from_json(json_path, b_id, args.num_parties)
            
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
            console.print("[bold cyan]Alice (party 0) compiling MatSAT solver before network sync...[/bold cyan]")
            from private_path_query_utils import compile_find_safe_path
            ok = await compile_find_safe_path(private_path_info=info, weighted=True)
            if not ok:
                console.print("[bold red]Alice compilation failed![/bold red]")
                sys.exit(1)

        # Sync before starting nodes!
        sync_port = args.port + 100
        if args.party_id == 0:
            server_task = asyncio.create_task(run_sync_server(sync_port, args.num_parties - 1))
            await asyncio.sleep(0.5)
            console.print("[dim]Alice (party 0) orchestrator ready. Waiting for Bobs...[/dim]")
            await server_task
        else:
            await wait_for_start(args.host, sync_port)

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
                compile_program=False,
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
            
            # Extract and display the actual path
            assignment = assignment_from_u_vector(
                u_vector=result.u_vector, T=T, V=V, symbols=syms
            )
            path = decode_path_from_assignment(assignment, T=T, V=V)
            console.print(f"\n[bold green]Extracted Path: {path}[/bold green]")
            
            # Save the integer path sequence
            out_path = args.out_path
            if not out_path and args.example:
                if args.example in ["diamond", "sat"]:
                    out_path = os.path.join(current_dir, "examples", args.example, "path.json")
                else: 
                    out_path = os.path.join(os.path.dirname(args.example), "path.json")
                    
            if out_path:
                try:
                    with open(out_path, 'w') as f:
                        json.dump(path, f)
                    console.print(f"[bold green]Saved abstract mathematical path sequence to {out_path}[/bold green]")
                except Exception as e:
                    console.print(f"[bold red]Failed to save path to {out_path}: {e}[/bold red]")
                    
            # 2. Automatically generate physical mapping if bounds are supplied
            if args.spatial_bounds and args.example:
                try:
                    import sys
                    sys.path.append(current_dir)
                    from distributed.graph_embedder import generate_spatial_embedding
                    
                    bounds_file = args.spatial_bounds
                    console.print(f"\n[bold cyan]Generating spatial embedding using boundaries from {bounds_file}...[/bold cyan]")
                    
                    # Compute mapping for the entire base graph
                    coords_map = generate_spatial_embedding(base_graph, bounds_file)
                    
                    # Write the generalized coordinates layout mapping
                    coords_out_path = os.path.join(os.path.dirname(out_path), "coordinates.json")
                    with open(coords_out_path, 'w') as f:
                        json.dump(coords_map, f, indent=2)
                    
                    # Convert the solved abstract path [0, 1, 2] -> physical waypoints [[lat, lon], ...]
                    physical_waypoints = []
                    for node_id in path:
                        pt = coords_map[str(node_id)]
                        physical_waypoints.append([pt["lat"], pt["lon"]])
                        
                    waypoints_out_path = os.path.join(os.path.dirname(out_path), "waypoints.json")
                    with open(waypoints_out_path, 'w') as f:
                        json.dump(physical_waypoints, f, indent=2)
                        
                    console.print(f"[bold green]Saved translated physical GPS coordinates to {waypoints_out_path}[/bold green]")
                    
                except Exception as e:
                    console.print(f"[bold red]Failed to generate or save physical mapping: {e}[/bold red]")

    elif args.mode == "verifier":
        # Note: verifier mode requires recompiling verifier program.
        # But this script uses join_computation directly.
        from private_path_query_utils import compile_verifier
        if args.party_id == 0:
            console.print("[bold cyan]Alice (party 0) compiling verifier before network sync...[/bold cyan]")
            # Recompile verifier if Alice
            await compile_verifier(args.num_parties, V, T, is_graph=True)
            
        sync_port = args.port + 100
        if args.party_id == 0:
            server_task = asyncio.create_task(run_sync_server(sync_port, args.num_parties - 1))
            await asyncio.sleep(0.5)
            console.print("[dim]Alice (party 0) orchestrator ready. Waiting for Bobs...[/dim]")
            await server_task
        else:
            await wait_for_start(args.host, sync_port)
            
        if args.party_id == 0:
            # Alice wants to verify the node path sequence from the example graph
            try:
                with open(json_path, 'r') as f:
                    example_data = json.load(f)
                
                # We expect the `alice_u_vector` to actually be an integer sequence representing the path
                # Ex: [0, 1, 2] represents start=0, move 0->1, move 1->2
                path_seq = example_data.get("alice_u_vector", [])
                
                if not path_seq or len(path_seq) < 2:
                    console.print("[bold red]Verifier mode requires 'alice_u_vector' integer sequence in the JSON example. Exiting.[/bold red]")
                    sys.exit(1)
            except Exception as e:
                console.print(f"[bold red]Failed to load alice_u_vector from {json_path}: {e}[/bold red]")
                sys.exit(1)
                
            # Convert node sequence to Verifier Edge pair inputs
            # format: start_node \n u1 \n v1 \n u2 \n v2 \n ...
            payload_lines = [str(path_seq[0])]
            for i in range(len(path_seq) - 1):
                payload_lines.append(str(path_seq[i]))
                payload_lines.append(str(path_seq[i + 1]))
                
            path_str_payload = "\n".join(payload_lines) + "\n"
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
            if args.example:
                # Use the loaded graph from the JSON example
                b_graph = bob_graph
            else:
                console.print("[bold red]Verifier mode currently requires an --example JSON configuration. Exiting.[/bold red]")
                sys.exit(1)
            
            console.print(f"[dim]Bob {args.party_id} inputs PRM partition graph for Verification. Edges: {[str(e) for e in b_graph.edges]}[/dim]")
            
            # For the verifier, Bob inputs a 1D array representing the flattened V*V adjacency matrix
            # where 2 = traversable, 1 = blocked, 0 = no edge.
            adj_list = b_graph.adjacency_list
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
    # Short-circuit networking/orchestration for execution-only mode
    if args.mode == "execute_waypoints":
        console.rule(f"[bold magenta]Starting Waypoint Execution[/bold magenta]")
        # 3. Only the physically connected node (usually Alice, or invoked directly) executes ways
        if args.party_id is not None and args.party_id != 0:
            console.print("[dim]Execution is handled by the primary connected node (party 0). Bobs idle...[/dim]")
            return

        out_path = args.out_path
        if not out_path and args.example:
            if args.example in ["diamond", "sat"]:
                waypoints_in_path = os.path.join(current_dir, "examples", args.example, "waypoints.json")
            else:
                waypoints_in_path = os.path.join(os.path.dirname(args.example), "waypoints.json")

        try:
            with open(waypoints_in_path, 'r') as f:
                physical_waypoints = json.load(f)
            console.print(f"[bold green]Loaded mapped waypoints successfully from {waypoints_in_path}[/bold green]")
        except FileNotFoundError:
            console.print(f"[bold red]Error: No previously computed waypoints.json found at {waypoints_in_path}. Run MatSAT solver first![/bold red]")
            return

        try:
            console.print("\n[bold cyan]Connecting to Surveyor ASV to execute waypoints...[/bold cyan]")
            # Append local submodule path if available
            sys.path.append(os.path.join(parent_dir, "searobotics_surveyor"))
            from surveyor_lib.surveyor import Surveyor

            host = '192.168.0.50'
            port = 8003
            throttle = 60

            # Read ERP from spatial bounds or default
            erp = tuple(physical_waypoints[0])
            if args.spatial_bounds:
                try:
                    with open(args.spatial_bounds, 'r') as f:
                        bounds = json.load(f)
                    erp = tuple(bounds.get("erp", physical_waypoints[0]))
                except Exception as e:
                    console.print(f"[bold yellow]Failed to read ERP from bounds, defaulting to first waypoint: {e}[/bold yellow]")

            boat = Surveyor(host=host, port=port, sensors_to_use=[], record=False)
            with boat:
                console.print("[bold green]Surveyor Connected![/bold green]")
                boat.send_waypoints(waypoints=physical_waypoints, erp=erp, throttle=throttle)
                console.print(f"Uploaded {len(physical_waypoints)} waypoints. ERP set to {erp}")

                console.print("[bold cyan]Starting Mission on Robot...[/bold cyan]")
                boat.set_waypoint_mode()

                import time
                while True:
                    state = boat.get_state()
                    mode = state.get('Control Mode', 'Unknown')
                    if mode == "Standby":
                        console.print("\n[bold green]Waypoint mission completed (Returned to Standby)![/bold green]")
                        break
                    time.sleep(2)

        except Exception as e:
            console.print(f"[bold red]Surveyor execution failed: {e}[/bold red]")

    console.rule(f"[bold magenta]Starting {args.role.capitalize()} (Party {args.party_id}) - Mode: {args.mode}[/bold magenta]")

    # 2. Execution and Inline Orchestration
    await execute_query(args)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed MatSat/Verifier Run")
    parser.add_argument("--role", choices=["alice", "bob"], help="Role to play")
    parser.add_argument("--party_id", type=int, help="MPC Player ID (Alice=0, Bob=1..N)")
    parser.add_argument("--num_parties", type=int, help="Total number of parties")
    parser.add_argument("--host", help="Host IP/Hostname of Alice (Coordinator)")
    parser.add_argument("--port", type=int, default=5000, help="Base Port to use (MP-SPDZ default 5000)")
    parser.add_argument("--protocol", choices=["shamir", "mascot"], default="mascot", help="MPC Protocol")
    parser.add_argument("--mode", choices=["matsat", "verifier", "execute_waypoints"], default="matsat", help="Action to perform")
    parser.add_argument("--example", default=None, help="Path to JSON predefined graph example, or 'diamond'/'sat'")
    parser.add_argument("--out_path", default=None, help="Optional JSON path to save the completed graph traversal path")
    parser.add_argument("--spatial_bounds", default=None, help="Optional bounding config JSON to transpile path directly into GPS coordinates")
    
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

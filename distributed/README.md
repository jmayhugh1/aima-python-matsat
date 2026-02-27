# Distributed PRM Benchmarker

This directory contains the tools for running distributed PRM (Probabilistic Roadmap) pathfinding benchmarks using MP-SPDZ.

## Prerequisites

- **Python Dependencies**: `matplotlib`, `cartopy` (for visualization), `rich`, `asyncio`.
- **MP-SPDZ**: Must be compiled and available in the parent directory.
- **uv**: Recommended for running scripts and managing dependencies.

## Standard Execution (2 Parties)

To run a PRM query, you need two terminals. Alice acts as the orchestrator/coordinator, and Bob acts as the ground-truth provider.

### Terminal 1: Alice (Party 0)
```bash
PYTHONPATH=. uv run python3 distributed/run_prm.py \
    --role alice \
    --party_id 0 \
    --num_parties 2 \
    --mode iterative_verifier \
    --graph_name 10v_25e \
    --spatial_bounds distributed/spatial_bounds.json \
    --min_dist 8.0 \
    --visualize
```

### Terminal 2: Bob (Party 1)
```bash
PYTHONPATH=. uv run python3 distributed/run_prm.py \
    --role bob \
    --party_id 1 \
    --num_parties 2 \
    --mode iterative_verifier \
    --graph_name 10v_25e \
    --host localhost \
    --spatial_bounds distributed/spatial_bounds.json
```

## Troubleshooting: Port Collisions

If you see `OSError: [Errno 98] address already in use`, run the following to clear lingering processes:

```bash
# Clear PRM sync and MP-SPDZ ports
lsof -i :5000 :5100 :5101 -t | xargs -r kill -9

# Terminate lingering party processes
pkill -f mascot-party.x
pkill -f run_prm.py
```

## Arguments

- `--mode`: 
  - `iterative_verifier`: Alice discovers the graph lazily through MPC queries.
  - `matsat_solver`: Alice finds a safe path assuming her current knowledge.
- `--graph_name`: Topology choice (e.g. `5v_6e`, `10v_25e`).
- `--spatial_bounds`: Path to `spatial_bounds.json` for GPS mapping.
- `--visualize`: Generates a satellite map overlay.
- `--min_dist`: Minimum distance in meters between nodes (prevents clustering).

## Output Files

Results are saved to `distributed/examples/<graph_name>/`:
- `coordinates.json`: Mapping of graph nodes to `[lat, lon]`.
- `waypoints.json`: The final safe path as a sequence of GPS coordinates.
- `embedding.png`: Satellite visualization showing the graph, fountains, boundary box, and confirmed path.

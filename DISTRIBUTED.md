# Distributed Bayesien Paths

This guide explains how to run the distributed version of the Bayesian Path query system locally (or across network). It supports both **MASCOT** (2-party malicious security) and **SHAMIR** (3-party information-theoretic security).

---

## Prerequisites

Ensure you have followed `INSTALLATION.md` and compiled the necessary protocols:

```bash
cd MP-SPDZ
make -j8 mascot-party.x shamir-party.x
cd ..
```

---

## Key Concepts

-   **Base Port**: The common port (e.g., 5000) that all parties agree upon. MP-SPDZ uses this to derive per-player offsets.
-   **Path Sharing Port**: An auxiliary TCP port used by Alice to broadcast the path to Bobs (so they can update their belief maps). This is automatically set to `Base Port + 100` (e.g., 5100).
-   **Role**: Each process plays either `alice` (Party 0) or `bob` (Party 1..N).

---

## Grid Generation Options

The system supports three methods for generating Bob's grid:

### 1. Hardcoded 3x3 Map (Default)
When using `--grid_size 3` without a seed, Bob uses a predefined traversable map:
```
0 0 1
1 0 1
1 0 0
```

### 2. Shared Seed Partitioned Maze (Recommended)
Use `--seed <INT>` to enable distributed maze generation:
- All Bobs use the same seed to deterministically generate an identical global maze
- Each Bob receives a unique subset of the maze walls (partitioned by coordinate hashing)
- The union of all Bob grids reconstructs the complete traversable maze
- **No central coordination required** - each Bob independently computes their partition

**Example**: With `--seed 999 --grid_size 5`, Bob 1 might see walls at positions (0,1), (1,3), while Bob 2 sees walls at (0,3), (2,1), etc. Together they form a complete maze.

### 3. Simple Hazard Pattern (Fallback)
For other grid sizes without a seed, Bob generates a simple grid with hazards at the center and position (1,0).

---

## 1. Running with MASCOT (2 Parties)

MASCOT is the default protocol. It requires exactly 2 parties: 1 Alice + 1 Bob.

### Terminal 1: Alice (Party 0)
```bash
python3 distributed_bayesien_paths.py --role alice --party_id 0 --num_parties 2 --host localhost --port 5000 --seed 123 --grid_size 5 --iterations 5
```

### Terminal 2: Bob (Party 1)
```bash
python3 distributed_bayesien_paths.py --role bob --party_id 1 --num_parties 2 --host localhost --port 5000 --seed 123 --grid_size 5 --iterations 5
```

> **Note**: Both must use `--port 5000` and the **same `--seed`** value.

---

## 2. Running with SHAMIR (3 Parties)

Shamir Secret Sharing requires at least 3 parties: 1 Alice + 2 Bobs.

### Terminal 1: Alice (Party 0)
```bash
python3 distributed_bayesien_paths.py --role alice --party_id 0 --num_parties 3 --host localhost --port 5000 --seed 999 --grid_size 5 --iterations 5 --protocol shamir
```

### Terminal 2: Bob 1 (Party 1)
```bash
python3 distributed_bayesien_paths.py --role bob --party_id 1 --num_parties 3 --host localhost --port 5000 --seed 999 --grid_size 5 --iterations 5 --protocol shamir
```

### Terminal 3: Bob 2 (Party 2)
```bash
python3 distributed_bayesien_paths.py --role bob --party_id 2 --num_parties 3 --host localhost --port 5000 --seed 999 --grid_size 5 --iterations 5 --protocol shamir
```

> **Important**: All parties must use the **same `--seed`** and `--grid_size` to ensure the global map is coherent.

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--role` | Yes | - | `alice` or `bob` |
| `--party_id` | Yes | - | MPC Player ID (Alice=0, Bob=1..N) |
| `--num_parties` | Yes | - | Total number of parties |
| `--host` | Yes | - | Host IP/Hostname (use `localhost` for local testing) |
| `--port` | No | 5000 | Base port for MPC communication |
| `--grid_size` | No | 3 | Grid dimensions (NxN) |
| `--path_length` | No | 3 | Length of query path |
| `--iterations` | No | 5 | Number of query iterations |
| `--protocol` | No | `mascot` | MPC protocol (`mascot` or `shamir`) |
| `--seed` | No | None | Shared seed for deterministic maze generation |

---

## Troubleshooting

-   **Connection Refused on 5100**: Ensure Alice is running. Bob will retry for ~20 seconds.
-   **MP-SPDZ Timeouts**: Ensure firewall allows traffic on ports `5000` through `5003` (and `5100`).
-   **Address Already in Use**: If a run crashes, previous processes might still be holding the ports. Kill them with `pkill -f python` or `pkill -f party.x`.
-   **Inconsistent Maps**: Ensure all Bobs use the **same `--seed`** and `--grid_size` values.

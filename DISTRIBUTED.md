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

## 1. Running with MASCOT (2 Parties)

MASCOT is the default protocol. It requires exactly 2 parties: 1 Alice + 1 Bob.

### Terminal 1: Alice (Party 0)
```bash
python3 distributed_bayesien_paths.py --role alice --party_id 0 --num_parties 2 --host localhost --port 5000 --iterations 5
```

### Terminal 2: Bob (Party 1)
```bash
python3 distributed_bayesien_paths.py --role bob --party_id 1 --num_parties 2 --host localhost --port 5000 --iterations 5
```

> **Note**: Both must use `--port 5000`.

---

## 2. Running with SHAMIR (3 Parties)

Shamir Secret Sharing requires at least 3 parties: 1 Alice + 2 Bobs.

### Terminal 1: Alice (Party 0)
```bash
python3 distributed_bayesien_paths.py --role alice --party_id 0 --num_parties 3 --host localhost --port 5000 --iterations 5 --protocol shamir
```

### Terminal 2: Bob 1 (Party 1)
```bash
python3 distributed_bayesien_paths.py --role bob --party_id 1 --num_parties 3 --host localhost --port 5000 --iterations 5 --protocol shamir
```

### Terminal 3: Bob 2 (Party 2)
```bash
python3 distributed_bayesien_paths.py --role bob --party_id 2 --num_parties 3 --host localhost --port 5000 --iterations 5 --protocol shamir
```

---

## Troubleshooting

-   **Connection Refused on 5100**: Ensure Alice is running. Bob will retry for ~20 seconds.
-   **MP-SPDZ Timeouts**: Ensure firewall allows traffic on ports `5000` through `5003` (and `5100`).
-   **Address Already in Use**: If a run crashes, previous processes might still be holding the ports. Kill them with `pkill -f python` or `pkill -f party.x`.

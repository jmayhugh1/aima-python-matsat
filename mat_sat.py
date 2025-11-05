from typing import *
from utils4e import Expr, expr
from logic4e import to_cnf, conjuncts, prop_symbols, dpll_satisfiable
import numpy as np
import cppimport
import pathlib
import subprocess
import os
import re
import time
import asyncio
import uuid
import shutil
import socket
import struct
import random
import threading

QMAT_DIR = "qmat-mat-encodings"

# Lock for port finding to avoid race conditions in concurrent scenarios
_port_finding_lock = threading.Lock()

# ---------- helpers: build instance matrices ----------


def _symbols_order(formula: Expr) -> List[Expr]:
    """Deterministic symbol order [x1, x2, ..., xn] (sorted by strng name)."""
    cnf = to_cnf(formula)
    return sorted(list(prop_symbols(cnf)), key=str)


def _clause_literals(cl: Expr) -> List[Expr]:
    s = str(to_cnf(cl))
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return [expr(tok) for tok in s.split(" | ")]


def _build_Q1_Q2(formula: Expr) -> Tuple[np.ndarray, np.ndarray, List[Expr]]:
    """
    Returns Q1 (pos literals), Q2 (neg literals) see https://arxiv.org/abs/2108.06481, and ordered symbols.
    Q1, Q2 shapes: (m, n) where m=num clauses, n=num vars.
    """
    cnf = to_cnf(formula)
    clauses = conjuncts(cnf)
    syms = _symbols_order(cnf)
    n, m = len(syms), len(clauses)
    Q1 = np.zeros((m, n), dtype=np.float64)
    Q2 = np.zeros((m, n), dtype=np.float64)
    for i, cl in enumerate(clauses):
        lits = _clause_literals(cl)
        for j, s in enumerate(syms):
            if s in lits:
                Q1[i, j] = 1.0
            elif expr("~" + str(s)) in lits:
                Q2[i, j] = 1.0
    return Q1, Q2, syms


# ---------- core: MatSat solver returning a dict ----------


def mat_sat(
    formula: Expr,
    ell: float = 1.0,
    max_try: int = 8,
    max_itr: int = 200,
    beta: float = 0.25,
    seed: int = 0,
) -> Dict[str, bool] | bool:  # i picked random numbers for these
    """
    Solve SAT via MatSat cost minimization.
    Returns: {symbol_name: bool, ...} if satisfiable, else {}.
    """
    Q1, Q2, syms = _build_Q1_Q2(formula)
    n = len(syms)
    if n == 0:
        return {}

    rng = np.random.default_rng(seed)
    ones_n = np.ones(n, dtype=np.float64)
    A = Q1 - Q2  # (m, n)
    Q2_ones = Q2 @ ones_n  # (m,)

    def _min1(x: np.ndarray) -> np.ndarray:
        return np.minimum(1.0, x)

    def _J(u: np.ndarray) -> float:
        # Jsat = sum(1 - min1(Q u_d)) + (ell/2)||u ⊙ (1-u)||^2
        c = Q2_ones + (A @ u)  # (m,)
        clause_pen = np.sum(1.0 - _min1(c))
        bin_term = u * (1.0 - u)
        return clause_pen + 0.5 * ell * float(np.dot(bin_term, bin_term))

    def _grad(u: np.ndarray) -> np.ndarray:
        # ∇J = (Q2 - Q1)^T * (Q u_d)<1 + ell * (u ⊙ (1-u) ⊙ (1-2u))
        c = Q2_ones + (A @ u)  # (m,)
        mask = (c < 1.0).astype(np.float64)  # indicator of unsatisfied clauses
        term1 = (Q2 - Q1).T @ mask  # (n,)
        d = u * (1.0 - u) * (1.0 - 2.0 * u)  # (n,)
        return term1 + ell * d

    def _unsat_count(u_bin: np.ndarray) -> int:
        # number of unsatisfied clauses under binary u
        c_b = Q2_ones + (A @ u_bin)
        return int(np.sum(1.0 - _min1(c_b)))

    # random initial relaxed assignment
    u = rng.uniform(0.0, 1.0, size=n)

    best_u_bin = None
    best_err = np.inf

    for _ in range(max_try):
        for _ in range(max_itr):
            J = _J(u)
            g = _grad(u)
            gnorm2 = float(np.dot(g, g))
            if gnorm2 == 0.0:
                break
            # adaptive step size alpha = J / ||∇J||^2
            u -= (J / gnorm2) * g
            np.clip(u, 0.0, 1.0, out=u)

            u_bin = (u >= 0.5).astype(np.float64)
            err = _unsat_count(u_bin)
            if err < best_err:
                best_err = err
                best_u_bin = u_bin.copy()
                if err == 0:
                    # Found satisfying assignment → return mapping
                    return {str(sym): bool(val) for sym, val in zip(syms, best_u_bin)}

        # randomized restart / noise mixing
        u = (1.0 - beta) * u + beta * rng.uniform(0.0, 1.0, size=n)

    # No satisfying assignment found
    return (
        False  # could not satisfy
        if best_err != 0
        else {str(sym): bool(val) for sym, val in zip(syms, best_u_bin)}
    )


def mat_sat_cpp(formula: Expr) -> dict[Expr, bool] | None:
    if type(formula) == Expr:
        formula = to_cnf(formula)
    if type(formula) != str:
        formula = str(formula)
        if len(formula) >= 2 and formula[0] == "(" and formula[-1] == ")":
            formula = formula[1:-1]
    m = cppimport.imp("matsat")
    assignment = m.mat_sat(formula, max_itr=2000)
    return {expr(key): val for key, val in assignment.items()} if assignment else None


def _find_consecutive_available_ports(
    num_parties: int, start_port: int = 5001, max_attempts: int = 100
) -> int:
    """Find a range of consecutive available ports.

    Uses a lock to serialize port finding in concurrent scenarios to avoid race conditions.

    Args:
        num_parties: Number of consecutive ports needed
        start_port: Starting port number to search from
        max_attempts: Maximum number of attempts to find ports

    Returns:
        Base port number where all consecutive ports are available

    Raises:
        RuntimeError: If no available port range found
    """
    # Use lock to serialize port finding and avoid race conditions
    with _port_finding_lock:
        # Add randomization to reduce conflicts between concurrent calls
        # Use process ID, time, and random to make it more unique per call
        random_offset = (
            (os.getpid() % 1000)
            + (int(time.time() * 1000) % 1000)
            + random.randint(0, 999)
        )

        for attempt in range(max_attempts):
            # Use large spacing (2000) and add randomization to avoid conflicts
            # This ensures concurrent calls are very unlikely to pick the same range
            test_port = start_port + (attempt * 2000) + (random_offset % 1000)
            # Ensure we stay in reasonable port range (avoid system ports < 1024 and high ports)
            test_port = max(5001, min(test_port, 65500 - num_parties))

            sockets = []
            try:
                # Try to bind to all consecutive ports we need
                for i in range(num_parties):
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                    try:
                        s.bind(("127.0.0.1", test_port + i))
                        sockets.append(s)
                    except OSError:
                        # Port not available, close all sockets and try next base port
                        for sock in sockets:
                            sock.close()
                        sockets = []
                        break

                # If we successfully bound to all ports, we found our range
                if len(sockets) == num_parties:
                    port = test_port
                    # Close all the test sockets
                    for s in sockets:
                        s.close()
                    # Small delay to ensure ports are fully released
                    time.sleep(0.05)
                    return port
            except Exception:
                # Clean up any remaining sockets
                for s in sockets:
                    try:
                        s.close()
                    except:
                        pass

        raise RuntimeError(
            f"Could not find {num_parties} consecutive available ports after {max_attempts} attempts"
        )


def _parse_is_solved_from_output(output: str) -> bool:
    """Parse is_solved value from MP-SPDZ output.

    Args:
        output: Combined stdout and stderr from MP-SPDZ execution

    Returns:
        True if is_solved = 1, False if is_solved = 0

    Raises:
        RuntimeError: If is_solved value not found in output
    """
    match = re.search(
        r"\[party\s+\d+\]\s+is_solved\s*=\s*(\d+)|is_solved\s*=\s*(\d+)", output
    )
    if match:
        is_solved_value = int(match.group(1) or match.group(2))
        return bool(is_solved_value)
    raise RuntimeError(f"Could not find is_solved value in output: {output}")


def _validate_mpspdz_paths(
    qmat_dir: str,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Validate MP-SPDZ paths and return resolved paths.

    Args:
        qmat_dir: Directory containing q matrices

    Returns:
        Tuple of (mp_spdz_dir, qmat_path, run_script_path)

    Raises:
        FileNotFoundError: If required paths don't exist
    """
    mp_spdz_dir = pathlib.Path("MP-SPDZ").resolve()
    run_script = mp_spdz_dir / "run-parties-proc.py"
    qmat_path = pathlib.Path(qmat_dir).resolve()

    if not run_script.exists():
        raise FileNotFoundError(f"run-parties-proc.py not found: {run_script}")
    if not qmat_path.exists():
        raise FileNotFoundError(f"QMAT directory not found: {qmat_path}")

    return mp_spdz_dir, qmat_path, run_script


def compile_mpspdz(qmat_dir: str = QMAT_DIR) -> None:
    """Compiles the MP-SPDZ program"""
    mp_spdz_dir = pathlib.Path("MP-SPDZ").resolve()
    script_path = mp_spdz_dir / "Programs" / "Source" / "multiparty_matsat.py"
    qmat_path = pathlib.Path(qmat_dir).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"MP-SPDZ source not found: {script_path}")
    if not qmat_path.exists():
        raise FileNotFoundError(f"QMAT directory not found: {qmat_path}")

    # Set PYTHONPATH if needed
    env = os.environ.copy()
    env["PYTHONPATH"] = str(mp_spdz_dir) + (":" + env.get("PYTHONPATH", ""))

    # Use relative path when running from MP-SPDZ directory
    script_rel_path = pathlib.Path("Programs") / "Source" / "multiparty_matsat.py"

    result = subprocess.run(
        ["python3", str(script_rel_path), "-d", str(qmat_path)],
        cwd=str(mp_spdz_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}\n{result.stdout}")


def run_mpspdz(
    qmat_dir: str = QMAT_DIR, protocol: str = "mascot", timeout: int = 300
) -> bool:
    """Runs the MP-SPDZ program and parses is_solved from the output.

    Returns True if is_solved = 1, False if is_solved = 0.
    """
    mp_spdz_dir, qmat_path, run_script = _validate_mpspdz_paths(qmat_dir)

    # Run the script - it will start all parties and wait for them
    try:
        result = subprocess.run(
            ["python3", str(run_script), str(qmat_path), protocol, "matsat"],
            cwd=str(mp_spdz_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"MP-SPDZ execution timed out after {timeout} seconds. "
            "The computation may still be running."
        )

    # Parse output for is_solved
    output = result.stdout + result.stderr

    # Check for errors first
    if result.returncode != 0:
        raise RuntimeError(
            f"MP-SPDZ execution failed with return code {result.returncode}.\n"
            f"Stderr: {result.stderr}\n"
            f"Stdout: {result.stdout}"
        )

    # Parse is_solved from output
    try:
        return _parse_is_solved_from_output(output)
    except RuntimeError:
        # Re-raise with more context
        raise RuntimeError(
            f"Could not find is_solved value in output.\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )


def mat_sat_mpspdz(formulas: List[Expr]) -> dict[Expr, bool] | None:
    """Run MatSat using MP-SPDZ multi-party computation.

    Returns a dict mapping symbols to bool values if satisfiable, None if unsatisfiable.
    """
    if not formulas:
        return None

    # Collect all symbols from all formulas to ensure consistent dimensions
    all_symbols = set()
    for formula in formulas:
        cnf = to_cnf(formula)
        all_symbols.update(prop_symbols(cnf))

    # Sort symbols for consistent ordering
    syms = sorted(list(all_symbols), key=str)
    n = len(syms)

    if n == 0:
        return None

    # Build matrices using consistent symbol set
    Q_matrices = []
    for formula in formulas:
        Q1, Q2, formula_syms = _build_Q1_Q2(formula)
        formula_n = len(formula_syms)

        # Ensure Q1 and Q2 have n columns (pad with zeros for missing symbols)
        if formula_n < n:
            Q1_padded = np.zeros((Q1.shape[0], n), dtype=np.float64)
            Q2_padded = np.zeros((Q2.shape[0], n), dtype=np.float64)
            # Map formula symbols to full symbol set
            for i, sym in enumerate(formula_syms):
                j = syms.index(sym)
                Q1_padded[:, j] = Q1[:, i]
                Q2_padded[:, j] = Q2[:, i]
            Q1, Q2 = Q1_padded, Q2_padded

        # Append the q1 and q2 left to right (concatenate horizontally)
        Q_matrix = np.concatenate((Q1, Q2), axis=1)
        Q_matrices.append(Q_matrix)

    # Ensure QMAT_DIR exists
    qmat_path = pathlib.Path(QMAT_DIR)
    qmat_path.mkdir(exist_ok=True)

    # Write matrices to QMAT_DIR as integers
    # get_dims() reads space-separated rows for dimension checking
    # But sint.get_input_from() reads one integer per line during execution
    # So we write as space-separated rows for get_dims(), and the input will be
    # provided as one integer per line during execution
    for i, q_matrix in enumerate(Q_matrices):
        # Convert to integers and write as space-separated rows
        q_matrix_int = q_matrix.astype(np.int32)
        np.savetxt(f"{QMAT_DIR}/q{i}.qmat", q_matrix_int, delimiter=" ", fmt="%d")

    # Compile the MP-SPDZ program
    compile_mpspdz(QMAT_DIR)

    # Run the MP-SPDZ program and get is_solved result
    is_solved = run_mpspdz(QMAT_DIR)

    if not is_solved:
        return None  # UNSAT

    # If SAT, we need to extract the assignment from the output
    # For now, return an empty dict to indicate SAT but assignment not parsed
    # TODO: Parse the u[i] values from output to get actual assignment
    return {}


async def compile_mpspdz_async(qmat_dir: str) -> None:
    """Async version of compile_mpspdz."""
    mp_spdz_dir = pathlib.Path("MP-SPDZ").resolve()
    script_path = mp_spdz_dir / "Programs" / "Source" / "multiparty_matsat.py"
    qmat_path = pathlib.Path(qmat_dir).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"MP-SPDZ source not found: {script_path}")
    if not qmat_path.exists():
        raise FileNotFoundError(f"QMAT directory not found: {qmat_path}")

    # Count the number of qmat files to determine number of parties
    qmat_files = sorted(qmat_path.glob("q*.qmat"))
    num_parties = len(qmat_files)

    if num_parties < 1:
        raise ValueError(f"No qmat files found in {qmat_path}")
    # print(f"Number of parties: {num_parties}")

    # Set PYTHONPATH if needed
    env = os.environ.copy()
    env["PYTHONPATH"] = str(mp_spdz_dir) + (":" + env.get("PYTHONPATH", ""))

    # Set the number of parties in the environment if needed
    # MP-SPDZ uses COMPILER_PARTIES environment variable or defaults to 2
    env["COMPILER_PARTIES"] = str(num_parties)

    # Use relative path when running from MP-SPDZ directory
    script_rel_path = pathlib.Path("Programs") / "Source" / "multiparty_matsat.py"

    process = await asyncio.create_subprocess_exec(
        "python3",
        str(script_rel_path),
        "-d",
        str(qmat_path),
        cwd=str(mp_spdz_dir),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{stderr.decode()}\n{stdout.decode()}")


async def run_mpspdz_async(
    qmat_dir: str = QMAT_DIR,
    protocol: str = "mascot",
    timeout: int = 300,
    port: int | None = None,
) -> bool:
    """Async version of run_mpspdz.

    Automatically finds an available port to avoid conflicts with concurrent calls.

    Args:
        qmat_dir: Directory containing q matrices
        protocol: MP-SPDZ protocol to use
        timeout: Timeout in seconds
    """
    mp_spdz_dir, qmat_path, run_script = _validate_mpspdz_paths(qmat_dir)

    # Count the number of parties (qmat files) to determine how many ports we need
    qmat_files = sorted(qmat_path.glob("q*.qmat"))
    num_parties = len(qmat_files)

    # Use provided port or find one dynamically
    if port is None:
        port = _find_consecutive_available_ports(num_parties)

    # Run the script - it will start all parties and wait for them
    try:
        process = await asyncio.create_subprocess_exec(
            "python3",
            str(run_script),
            str(qmat_path),
            protocol,
            "matsat",
            "--port",
            str(port),
            cwd=str(mp_spdz_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"MP-SPDZ execution timed out after {timeout} seconds. "
            "The computation may still be running."
        )

    # Parse output for is_solved
    output = (stdout + stderr).decode()

    # Check for errors first
    if process.returncode != 0:
        raise RuntimeError(
            f"MP-SPDZ execution failed with return code {process.returncode}.\n"
            f"Stderr: {stderr.decode()}\n"
            f"Stdout: {stdout.decode()}"
        )

    # Parse is_solved from output
    try:
        return _parse_is_solved_from_output(output)
    except RuntimeError:
        # Re-raise with more context
        raise RuntimeError(
            f"Could not find is_solved value in output.\n"
            f"Stdout: {stdout.decode()}\n"
            f"Stderr: {stderr.decode()}"
        )


def _count_parties_for_formulas(formulas: List[Expr]) -> int:
    """Count the number of parties needed for a list of formulas.

    Each formula becomes one party in MP-SPDZ, so the number of parties
    is simply the number of formulas.

    Args:
        formulas: List of formulas

    Returns:
        Number of parties needed
    """
    return len(formulas) if formulas else 0


async def reserve_ports_for_formula_sets(
    formula_sets: List[List[Expr]], start_port: int = 5001
) -> List[int]:
    """Reserve ports upfront for multiple formula sets.

    This function calculates how many parties each formula set needs,
    then reserves consecutive port ranges for all of them. This avoids
    race conditions when running concurrent calls.

    Args:
        formula_sets: List of formula sets, where each set is a list of formulas
        start_port: Starting port number to search from

    Returns:
        List of base ports, one for each formula set
    """
    # Calculate number of parties needed for each formula set
    num_parties_list = [
        _count_parties_for_formulas(formulas) for formulas in formula_sets
    ]

    # Reserve all port ranges upfront using the lock
    reserved_ports = []
    current_base_port = start_port

    with _port_finding_lock:
        for num_parties in num_parties_list:
            if num_parties == 0:
                reserved_ports.append(None)
                continue

            # Find consecutive ports for this formula set
            port = _find_consecutive_available_ports_unlocked(
                num_parties, current_base_port
            )
            reserved_ports.append(port)
            # Move to next potential port range (add large spacing to avoid conflicts)
            current_base_port = port + num_parties + 1000

    return reserved_ports


def _find_consecutive_available_ports_unlocked(
    num_parties: int, start_port: int = 5001, max_attempts: int = 100
) -> int:
    """Find consecutive available ports without using the lock.

    This is used internally when we already have the lock from reserve_ports_for_formula_sets.

    Args:
        num_parties: Number of consecutive ports needed
        start_port: Starting port number to search from
        max_attempts: Maximum number of attempts to find ports

    Returns:
        Base port number where all consecutive ports are available
    """
    for attempt in range(max_attempts):
        test_port = start_port + (attempt * 100)
        # Ensure we stay in reasonable port range
        test_port = max(5001, min(test_port, 65500 - num_parties))

        sockets = []
        try:
            # Try to bind to all consecutive ports we need
            for i in range(num_parties):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
                try:
                    s.bind(("127.0.0.1", test_port + i))
                    sockets.append(s)
                except OSError:
                    # Port not available, close all sockets and try next base port
                    for sock in sockets:
                        sock.close()
                    sockets = []
                    break

            # If we successfully bound to all ports, we found our range
            if len(sockets) == num_parties:
                port = test_port
                # Keep sockets open - don't close them yet, we'll release when done
                # Store them for later cleanup
                for s in sockets:
                    s.close()
                time.sleep(0.01)
                return port
        except Exception:
            # Clean up any remaining sockets
            for s in sockets:
                try:
                    s.close()
                except:
                    pass

    raise RuntimeError(
        f"Could not find {num_parties} consecutive available ports after {max_attempts} attempts"
    )


async def mat_sat_mpspdz_async(
    formulas: List[Expr], base_qmat_dir: str = QMAT_DIR, port: int | None = None
) -> dict[Expr, bool] | None:
    """Async version of mat_sat_mpspdz that uses a temporary subfolder for each call.

    Creates a unique subfolder within base_qmat_dir, writes q matrices there,
    runs MP-SPDZ, and then cleans up the subfolder. Automatically finds
    an available port to avoid conflicts with concurrent calls, unless a port
    is provided (e.g., from reserve_ports_for_formula_sets).

    Args:
        formulas: List of formulas to solve
        base_qmat_dir: Base directory for q matrices (default: QMAT_DIR)
        port: Optional pre-reserved base port. If None, finds one automatically.

    Returns:
        A dict mapping symbols to bool values if satisfiable, None if unsatisfiable.
    """
    if not formulas:
        return None

    # Collect all symbols from all formulas to ensure consistent dimensions
    all_symbols = set()
    for formula in formulas:
        cnf = to_cnf(formula)
        all_symbols.update(prop_symbols(cnf))

    # Sort symbols for consistent ordering
    syms = sorted(list(all_symbols), key=str)
    n = len(syms)

    if n == 0:
        return None

    # Build matrices using consistent symbol set
    Q_matrices = []
    for formula in formulas:
        Q1, Q2, formula_syms = _build_Q1_Q2(formula)
        formula_n = len(formula_syms)

        # Ensure Q1 and Q2 have n columns (pad with zeros for missing symbols)
        if formula_n < n:
            Q1_padded = np.zeros((Q1.shape[0], n), dtype=np.float64)
            Q2_padded = np.zeros((Q2.shape[0], n), dtype=np.float64)
            # Map formula symbols to full symbol set
            for i, sym in enumerate(formula_syms):
                j = syms.index(sym)
                Q1_padded[:, j] = Q1[:, i]
                Q2_padded[:, j] = Q2[:, i]
            Q1, Q2 = Q1_padded, Q2_padded

        # Append the q1 and q2 left to right (concatenate horizontally)
        Q_matrix = np.concatenate((Q1, Q2), axis=1)
        Q_matrices.append(Q_matrix)

    # Create a unique temporary subfolder for this call
    base_path = pathlib.Path(base_qmat_dir)
    base_path.mkdir(exist_ok=True)

    # Create a unique subfolder using UUID
    unique_id = str(uuid.uuid4())
    temp_qmat_dir = base_path / unique_id
    temp_qmat_dir.mkdir(exist_ok=True)

    try:
        # Write matrices to the temporary subfolder as integers
        for i, q_matrix in enumerate(Q_matrices):
            q_matrix_int = q_matrix.astype(np.int32)
            qmat_file = temp_qmat_dir / f"q{i}.qmat"
            np.savetxt(str(qmat_file), q_matrix_int, delimiter=" ", fmt="%d")

        # Compile the MP-SPDZ program
        await compile_mpspdz_async(str(temp_qmat_dir))

        # Run the MP-SPDZ program and get is_solved result
        # Use provided port or let it find one automatically
        is_solved = await run_mpspdz_async(str(temp_qmat_dir), port=port)

        if not is_solved:
            return None  # UNSAT

        # If SAT, we need to extract the assignment from the output
        # For now, return an empty dict to indicate SAT but assignment not parsed
        # TODO: Parse the u[i] values from output to get actual assignment
        return {}
    finally:
        # Clean up the temporary subfolder
        if temp_qmat_dir.exists():
            shutil.rmtree(temp_qmat_dir)

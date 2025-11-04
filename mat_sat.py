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


QMAT_DIR = "qmat-mat-encodings"
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
    mp_spdz_dir = pathlib.Path("MP-SPDZ").resolve()
    run_script = mp_spdz_dir / "run-parties-proc.py"
    qmat_path = pathlib.Path(qmat_dir).resolve()

    if not run_script.exists():
        raise FileNotFoundError(f"run-parties-proc.py not found: {run_script}")
    if not qmat_path.exists():
        raise FileNotFoundError(f"QMAT directory not found: {qmat_path}")

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
    # run-parties-proc.py outputs: [party X] is_solved = Y
    output = result.stdout + result.stderr

    # Look for is_solved pattern in the output (handles both formats)
    # Match "[party X] is_solved = Y" or just "is_solved = Y"
    match = re.search(
        r"\[party\s+\d+\]\s+is_solved\s*=\s*(\d+)|is_solved\s*=\s*(\d+)", output
    )
    if match:
        is_solved_value = int(match.group(1) or match.group(2))
        return bool(is_solved_value)

    # If not found, check if there were errors
    if result.returncode != 0:
        raise RuntimeError(
            f"MP-SPDZ execution failed with return code {result.returncode}.\n"
            f"Stderr: {result.stderr}\n"
            f"Stdout: {result.stdout}"
        )

    # If we get here, we couldn't find is_solved in the output
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
    print(f"Number of parties: {num_parties}")

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
    port: int = 5001,
) -> bool:
    """Async version of run_mpspdz.

    Args:
        qmat_dir: Directory containing q matrices
        protocol: MP-SPDZ protocol to use
        timeout: Timeout in seconds
        port: Base port number for MP-SPDZ parties (default: 5001)
    """
    mp_spdz_dir = pathlib.Path("MP-SPDZ").resolve()
    run_script = mp_spdz_dir / "run-parties-proc.py"
    qmat_path = pathlib.Path(qmat_dir).resolve()

    if not run_script.exists():
        raise FileNotFoundError(f"run-parties-proc.py not found: {run_script}")
    if not qmat_path.exists():
        raise FileNotFoundError(f"QMAT directory not found: {qmat_path}")

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

    # Look for is_solved pattern in the output
    match = re.search(
        r"\[party\s+\d+\]\s+is_solved\s*=\s*(\d+)|is_solved\s*=\s*(\d+)", output
    )
    if match:
        is_solved_value = int(match.group(1) or match.group(2))
        return bool(is_solved_value)

    # If not found, check if there were errors
    if process.returncode != 0:
        raise RuntimeError(
            f"MP-SPDZ execution failed with return code {process.returncode}.\n"
            f"Stderr: {stderr.decode()}\n"
            f"Stdout: {stdout.decode()}"
        )

    # If we get here, we couldn't find is_solved in the output
    raise RuntimeError(
        f"Could not find is_solved value in output.\n"
        f"Stdout: {stdout.decode()}\n"
        f"Stderr: {stderr.decode()}"
    )


async def mat_sat_mpspdz_async(
    formulas: List[Expr], base_qmat_dir: str = QMAT_DIR, port: int | None = None
) -> dict[Expr, bool] | None:
    """Async version of mat_sat_mpspdz that uses a temporary subfolder for each call.

    Creates a unique subfolder within base_qmat_dir, writes q matrices there,
    runs MP-SPDZ, and then cleans up the subfolder.

    Args:
        formulas: List of formulas to solve
        base_qmat_dir: Base directory for q matrices (default: QMAT_DIR)
        port: Base port number for MP-SPDZ parties. If None, uses a port derived from UUID.

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

    # If port not provided, derive one from the UUID to avoid conflicts
    if port is None:
        # Use hash of UUID to get a port in range 5001-5999
        port = 5001 + (hash(unique_id) % 999)

    try:
        # Write matrices to the temporary subfolder as integers
        for i, q_matrix in enumerate(Q_matrices):
            q_matrix_int = q_matrix.astype(np.int32)
            qmat_file = temp_qmat_dir / f"q{i}.qmat"
            np.savetxt(str(qmat_file), q_matrix_int, delimiter=" ", fmt="%d")

        # Compile the MP-SPDZ program
        await compile_mpspdz_async(str(temp_qmat_dir))

        # Run the MP-SPDZ program and get is_solved result
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

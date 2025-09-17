# dimacs_parser.py

from typing import List, Dict, Any, Optional


def parse_dimacs_from_file(path: str) -> Dict[str, Any]:
    """Parse a DIMACS CNF from a file path."""
    content = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Error reading DIMACS file {path}: {e}")
    dimacs_text: str = parse_dimacs_text(content)
    return dimacs_text


def parse_dimacs_text(text: str) -> Dict[str, Any]:
    """
    Parse a DIMACS CNF string.

    Returns:
        {
            "num_vars": int,
            "num_clauses": int,
            "clauses": List[List[int]],  # each literal is a nonzero int; clause ends at 0
        }
    """
    num_vars = 0
    num_clauses = 0
    clauses: List[List[int]] = []
    cur: List[int] = []

    lines = text.splitlines()
    stop = False

    for raw in lines:
        if stop:
            break
        line = raw.strip()
        if not line:
            continue
        if line[0] in ("c", "C"):
            # comment line: skip
            continue
        if line[0] in ("p", "P"):
            # problem header: p cnf <vars> <clauses>
            parts = line.split()
            if len(parts) < 4 or parts[1].lower() != "cnf":
                raise ValueError(f"Malformed DIMACS header: {line}")
            num_vars = int(parts[2])
            num_clauses = int(parts[3])
            continue
        if line[0] == "%":
            # end-of-instance marker (common in some generators)
            break

        for tok in line.split():
            if tok == "%":
                stop = True
                break
            try:
                lit = int(tok)
            except ValueError:
                raise ValueError(f"Non-integer token in DIMACS: {tok!r}")
            if lit == 0:
                # end of current clause
                if not cur:
                    raise ValueError("Encountered empty clause (unsatisfiable).")
                clauses.append(cur)
                cur = []
            else:
                cur.append(lit)

    # Some files omit the final trailing 0 for the last clause; accept it.
    if cur:
        clauses.append(cur)

    # Fill header if missing
    if num_vars == 0:
        num_vars = max((abs(l) for cl in clauses for l in cl), default=0)
    if num_clauses == 0:
        num_clauses = len(clauses)

    return {
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "clauses": clauses,
    }


def parse_dimacs_file(path: str) -> Dict[str, Any]:
    """Parse a DIMACS CNF from a file path."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_dimacs_text(f.read())


def dimacs_to_cnf_string(
    dimacs: Dict[str, Any], var_prefix: str = "X", var_names: Optional[List[str]] = None
) -> str:
    """
    Convert parsed DIMACS to your CNF string style:
      (~X1 | X3) & (X2 | ~X4 | X7)

    Args:
        dimacs: dict returned by parse_dimacs_text / parse_dimacs_file
        var_prefix: used if var_names is None (e.g., 'X' -> X1..Xn)
        var_names: optional list of names (1-based indexing -> var_names[i-1])

    Returns:
        CNF string suitable for expr("...") or your C++ parser.
    """
    clauses: List[List[int]] = dimacs["clauses"]

    def lit_str(lit: int) -> str:
        neg = lit < 0
        v = abs(lit)
        if var_names is not None:
            if not (1 <= v <= len(var_names)):
                raise ValueError(
                    f"Variable index {v} out of range for var_names (len={len(var_names)})."
                )
            name = var_names[v - 1]
        else:
            name = f"{var_prefix}{v}"
        return f"{'~' if neg else ''}{name}"

    clause_strs = []
    for cl in clauses:
        if not cl:
            raise ValueError("Empty clause cannot be rendered.")
        clause_strs.append("(" + " | ".join(lit_str(l) for l in cl) + ")")
    return " & ".join(clause_strs)


# --- quick demo ---
if __name__ == "__main__":
    file_name : str = "./dimacs_cnf/uf20-0999.cnf"
    dimacs = parse_dimacs_file(file_name)
    print(f"Parsed DIMACS: {dimacs['num_vars']} vars, {dimacs['num_clauses']} clauses.")
    print(dimacs_to_cnf_string(dimacs, var_prefix="X"))

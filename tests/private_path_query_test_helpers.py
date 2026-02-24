import asyncio
import inspect
import os

from logic4e import to_cnf, conjuncts, disjuncts
from utils4e import Expr

from private_path_query_logic import (
    force_pos_const,
    physics,
    bob_physics,
    alice_physics,
    build_physics_q,
    _build_bob_q,
    _build_alice_q,
    directed_pairs_from_edges,
    ordered_symbols,
    assignment_from_u_vector,
    decode_path_from_assignment,
    SOFT_CLAUSE_WEIGHT,
    compute_hard_clause_weight,
)
from private_path_query_utils import (
    Graph,
    GraphPath,
    Vertex,
    Edge,
    EdgeState,
    PrivatePathInfo,
    ComputationResult,
    compile_find_safe_path,
    join_computation_find_safe_path,
)


def _and_all(clauses):
    assert clauses, "Expected at least one clause"
    return clauses[0] if len(clauses) == 1 else Expr("&", *clauses)


def _force_graph_path_moves(path: GraphPath):
    """Constrain position bits to follow the concrete vertex sequence in GraphPath."""
    if not path.moves:
        return []
    vertices = [path.start.id] + [edge.vertex2.id for edge in path.moves]
    max_vertex = max(vertices)
    b = 1 if max_vertex <= 1 else max_vertex.bit_length()
    clauses = []
    for t, vertex_id in enumerate(vertices):
        clauses.extend(force_pos_const(t=t, v=vertex_id, b=b))
    return clauses


def _print_sat_moves(model, T: int, V: int):
    path = decode_path_from_assignment(model, T=T, V=V)
    timeline = [(t, f"POS={path[t]}") for t in range(len(path))]
    print(f"SAT position timeline: {timeline}")


def _print_assignments_from_u_vector(
    u_vector: list[int], T: int, V: int, symbols: list[Expr] | None = None
):
    model = assignment_from_u_vector(u_vector=u_vector, T=T, V=V, symbols=symbols)
    true_symbols = sorted(str(sym) for sym, val in model.items() if val)
    print(f"Decoded TRUE assignments: {true_symbols}")
    _print_sat_moves(model, T, V)


def _verify_result_and_print(
    result: ComputationResult | None,
    *,
    expect_sat: bool,
    T: int,
    V: int,
    symbols: list[Expr] | None = None,
    start: int | None = None,
    goal: int | None = None,
    domain_edges: list[Edge] | None = None,
    bob_edges_by_party: list[list[Edge]] | None = None,
):
    if result is None:
        return
    if result.u_vector is not None:
        _print_assignments_from_u_vector(result.u_vector, T=T, V=V, symbols=symbols)
        # Show clause reassembly with violated clauses if we have the necessary info
        if (
            start is not None
            and goal is not None
            and domain_edges is not None
            and bob_edges_by_party is not None
            and all(v in (0, 1) for v in result.u_vector)
        ):
            _print_clause_reassembly_from_u_vector(
                u_vector=result.u_vector,
                T=T,
                V=V,
                start=start,
                goal=goal,
                domain_edges=domain_edges,
                bob_edges_by_party=bob_edges_by_party,
            )
    assert bool(result.is_solved) is bool(expect_sat)
    assert result.information_gain == 0.0


def _resolve_mode(mode: str | None = None) -> str | None:
    selected = mode if mode is not None else os.getenv("FIND_SAFE_PATH_MODE")
    if selected is None or str(selected).strip() == "":
        return None
    selected = str(selected).strip().lower()
    if selected == "full":
        selected = "baseline"
    if selected not in {"baseline", "optimized", "both"}:
        raise ValueError(
            "mode must be one of: baseline/full, optimized, both, or empty"
        )
    return selected


def _unknown_domain_edges_from_graph(graph: Graph) -> list[Edge]:
    return [
        Edge(e.vertex1, e.vertex2, EdgeState.UNKNOWN) for e in graph.to_directed_edges()
    ]


def _detect_test_name() -> str | None:
    """Walk call stack to find the first function starting with 'test_'."""
    for frame_info in inspect.stack():
        if frame_info.function.startswith("test_"):
            return frame_info.function
    return None


def _make_private_path_info(
    T: int,
    V: int,
    num_bobs: int = 1,
    use_edge_domain: bool = False,
    edge_domain_edges: list[Edge] | None = None,
    weighted: bool = True,
) -> PrivatePathInfo:
    """
    Create a PrivatePathInfo with auto-calculated row counts.

    Args:
        T: Time horizon
        V: Number of vertices
        num_bobs: Number of Bob parties (default 1)
        use_edge_domain: Whether to use edge domain optimization
        edge_domain_edges: Edges defining the domain (required if use_edge_domain=True)
        weighted: Whether to use weighted clauses

    Returns:
        PrivatePathInfo with correctly calculated rows_per_id
    """
    num_parties = num_bobs + 1  # Alice + Bobs

    compact_pairs = None
    if use_edge_domain and edge_domain_edges:
        compact_pairs = directed_pairs_from_edges(edge_domain_edges, V)

    syms = ordered_symbols(T, V, directed_pairs=compact_pairs)

    # Calculate row counts using internal helpers
    q_alice, _ = _build_alice_q(start=0, goal=0, T=T, V=V, symbols=syms)
    q_bob, _ = _build_bob_q(
        edges=[], T=T, V=V, symbols=syms, directed_pairs=compact_pairs
    )
    q_physics, _ = build_physics_q(T=T, V=V, symbols=syms, directed_pairs=compact_pairs)

    alice_rows = int(q_physics.shape[0] + q_alice.shape[0])
    bob_rows = int(q_bob.shape[0])

    return PrivatePathInfo(
        num_parties=num_parties,
        T=T,
        V=V,
        rows_per_id=[alice_rows] + [bob_rows] * num_bobs,
        use_weight_vector=weighted,
        use_edge_domain=use_edge_domain,
        edge_domain_edges=edge_domain_edges,
    )


async def _run_find_safe_path_helper(
    info: PrivatePathInfo,
    graphs: list[Graph],
    start: Vertex,
    goal: Vertex,
    port: int = 5003,
    weighted: bool = True,
    test_name: str | None = None,
) -> ComputationResult:
    """
    Run find_safe_path with PrivatePathInfo.

    Args:
        info: PrivatePathInfo containing T, V, edge_domain, rows_per_id, etc.
        graphs: List of graphs for each Bob party
        start: Start vertex (Alice's private input)
        goal: Goal vertex (Alice's private input)
        port: Port for MPC communication
        weighted: Whether to use weighted clauses
        test_name: Optional test name for logging
    """
    # Auto-detect test name from call stack if not provided
    if test_name is None:
        test_name = _detect_test_name()

    # Validate info matches graphs
    assert (
        info.num_parties == len(graphs) + 1
    ), f"PrivatePathInfo.num_parties={info.num_parties} but got {len(graphs)} graphs (expected {info.num_parties - 1})"

    # Calculate columns: 2 * number of symbols
    directed_pairs = info.edge_domain_pairs() if info.use_edge_domain else None
    syms = ordered_symbols(info.T, info.V, directed_pairs=directed_pairs)
    num_columns = 2 * len(syms)

    print(f"Total Columns in Q Matrix: {num_columns} (symbols={len(syms)})")
    print(f"Total Rows in Q matrix: {sum(info.rows_per_id)}")

    ok = await compile_find_safe_path(private_path_info=info, weighted=weighted)
    assert ok

    tasks = [
        asyncio.create_task(
            join_computation_find_safe_path(
                id=0,
                private_path_info=info,
                start=start,
                goal=goal,
                compile_program=False,
                port=port,
                weighted=weighted,
                test_name=test_name,
            )
        )
    ]
    for i, graph in enumerate(graphs, start=1):
        tasks.append(
            asyncio.create_task(
                join_computation_find_safe_path(
                    id=i,
                    private_path_info=info,
                    graph=graph,
                    compile_program=False,
                    port=port,
                    weighted=weighted,
                    test_name=test_name,
                )
            )
        )

    results = await asyncio.gather(*tasks)
    return results[0]


async def _run_find_safe_path_compare_modes(
    info: PrivatePathInfo,
    graphs: list[Graph],
    start: Vertex,
    goal: Vertex,
    port: int,
    mode: str | None = None,
    weighted: bool = True,
    test_name: str | None = None,
) -> tuple[ComputationResult | None, ComputationResult | None]:
    """
    Run find_safe_path comparing baseline vs optimized modes.

    Args:
        info: PrivatePathInfo containing T, V, edge_domain, etc.
        graphs: List of graphs for each Bob party
        start: Start vertex
        goal: Goal vertex
        port: Base port for MPC communication
        mode: "baseline", "optimized", or "both" (auto-detected if None)
        weighted: Whether to use weighted clauses
        test_name: Optional test name for logging
    """
    # Auto-detect test name from call stack if not provided
    if test_name is None:
        test_name = _detect_test_name()

    selected_mode = _resolve_mode(mode)
    baseline: ComputationResult | None = None
    optimized: ComputationResult | None = None

    auto_mode = "optimized" if info.use_edge_domain else "baseline"
    effective_mode = selected_mode or auto_mode

    if effective_mode in ("baseline", "both"):
        # Create baseline info (no edge domain)
        baseline_info = _make_private_path_info(
            T=info.T,
            V=info.V,
            num_bobs=len(graphs),
            use_edge_domain=False,
            weighted=weighted,
        )
        baseline = await _run_find_safe_path_helper(
            info=baseline_info,
            graphs=graphs,
            start=start,
            goal=goal,
            port=port,
            weighted=weighted,
            test_name=test_name,
        )
    if effective_mode in ("optimized", "both"):
        # Use provided info or create optimized info
        optimized_info = (
            info
            if info.use_edge_domain
            else _make_private_path_info(
                T=info.T,
                V=info.V,
                num_bobs=len(graphs),
                use_edge_domain=True,
                edge_domain_edges=info.edge_domain_edges
                or _unknown_domain_edges_from_graph(graphs[0]),
                weighted=weighted,
            )
        )
        optimized = await _run_find_safe_path_helper(
            info=optimized_info,
            graphs=graphs,
            start=start,
            goal=goal,
            port=port + 1,
            weighted=weighted,
            test_name=test_name,
        )

    print(
        "MODE_COMPARE "
        f"mode={effective_mode} "
        f"baseline(solved={None if baseline is None else baseline.is_solved}, sat={None if baseline is None else baseline.satisfied_clauses}) "
        f"optimized(solved={None if optimized is None else optimized.is_solved}, sat={None if optimized is None else optimized.satisfied_clauses})"
    )
    return baseline, optimized


def _is_soft_allowed_clause(clause) -> bool:
    return (
        clause.op == "~"
        and len(clause.args) == 1
        and clause.args[0].op == "Allowed"
        and len(clause.args[0].args) == 2
    )


def _evaluate_clause(clause, assignment) -> bool:
    for lit in disjuncts(clause):
        if lit.op == "~":
            sym = lit.args[0]
            lit_value = not assignment.get(sym, False)
        else:
            lit_value = assignment.get(lit, False)
        if lit_value:
            return True
    return False


def _print_clause_reassembly_from_u_vector(
    *,
    u_vector: list[int],
    T: int,
    V: int,
    start: int,
    goal: int,
    domain_edges: list[Edge],
    bob_edges_by_party: list[list[Edge]],
    num_parties: int | None = None,
):
    compact_pairs = directed_pairs_from_edges(domain_edges, V)
    symbols = ordered_symbols(T, V, directed_pairs=compact_pairs)
    assignment = assignment_from_u_vector(u_vector, T=T, V=V, symbols=symbols)

    block_formulas: list[tuple[str, list]] = [
        ("physics", physics(T=T, V=V, directed_pairs=compact_pairs)),
        ("alice", alice_physics(start=start, goal=goal, T=T, V=V)),
    ]
    for party_offset, bob_edges in enumerate(bob_edges_by_party, start=1):
        bob_formulas, _ = bob_physics(
            edges=bob_edges, V=V, directed_pairs=compact_pairs
        )
        block_formulas.append(
            (
                f"bob[{party_offset}]",
                bob_formulas,
            )
        )

    total = 0
    total_violated = 0
    total_weighted_satisfied = 0.0
    violated_rows: list[str] = []
    violated_allowed_rows: list[str] = []

    # Compute hard clause weight dynamically based on number of parties
    effective_num_parties = num_parties if num_parties else len(bob_edges_by_party) + 1
    hard_weight = compute_hard_clause_weight(effective_num_parties)

    print("=== Clause reassembly from u_vector ===")
    for block_name, formulas in block_formulas:
        block_total = 0
        block_violated = 0
        for formula_idx, formula in enumerate(formulas):
            for local_clause_idx, clause in enumerate(conjuncts(to_cnf(formula))):
                sat = _evaluate_clause(clause, assignment)
                weight = (
                    SOFT_CLAUSE_WEIGHT
                    if _is_soft_allowed_clause(clause)
                    else hard_weight
                )
                total += 1
                block_total += 1
                if sat:
                    total_weighted_satisfied += float(weight)
                else:
                    total_violated += 1
                    block_violated += 1
                    row_msg = (
                        f"[{block_name}] formula[{formula_idx}] cnf[{local_clause_idx}] "
                        f"weight={weight}: {clause}"
                    )
                    violated_rows.append(row_msg)
                    if "Allowed(" in str(clause):
                        violated_allowed_rows.append(row_msg)
        print(f"  block={block_name} total={block_total} violated={block_violated}")

    print(
        f"Reassembled totals: rows={total}, violated={total_violated}, "
        f"weighted_satisfied={total_weighted_satisfied}"
    )
    print(f"Violated clauses ({len(violated_rows)}):")
    for row in violated_rows:
        print(f"  {row}")
    print(f"Violated Allowed-related clauses ({len(violated_allowed_rows)}):")
    for row in violated_allowed_rows:
        print(f"  {row}")


def _assert_optimized_result(
    optimized,
    expect_sat: bool,
    T: int,
    V: int,
    start: int,
    goal: int,
    domain_edges: list[Edge],
    bob_edges: list[Edge] | None = None,
    bob_edges_by_party: list[list[Edge]] | None = None,
    print_cnf: bool = False,
):
    assert optimized is not None
    compact_pairs = directed_pairs_from_edges(domain_edges, V)
    optimized_syms = ordered_symbols(T, V, directed_pairs=compact_pairs)

    # Always print diagnostics before assertions so failures keep context.
    has_u = optimized.u_vector is not None
    u_is_binary = bool(has_u and all(v in (0, 1) for v in (optimized.u_vector or [])))
    if has_u:
        if u_is_binary:
            _print_assignments_from_u_vector(
                optimized.u_vector, T=T, V=V, symbols=optimized_syms
            )
        else:
            print(
                "WARNING: Non-binary u_vector encountered; "
                "skipping decoded assignment print."
            )
            print(f"u_vector sample={optimized.u_vector[:8]}")

    if has_u and (bob_edges is not None or bob_edges_by_party):
        bob_payloads = (
            bob_edges_by_party
            if bob_edges_by_party is not None
            else [bob_edges] if bob_edges is not None else []
        )
        if print_cnf:
            print("=== build_physics_q: Physics payload ===")
            build_physics_q(
                T=T,
                V=V,
                symbols=optimized_syms,
                directed_pairs=compact_pairs,
                print_cnf_clauses=True,
            )
            print("=== _build_alice_q: Alice payload ===")
            _build_alice_q(
                start=start,
                goal=goal,
                T=T,
                V=V,
                symbols=optimized_syms,
                print_cnf_clauses=True,
            )
            for party_id, party_bob_edges in enumerate(bob_payloads, start=1):
                print(f"=== _build_bob_q: Bob party {party_id} payload ===")
                _build_bob_q(
                    edges=party_bob_edges,
                    T=T,
                    V=V,
                    symbols=optimized_syms,
                    directed_pairs=compact_pairs,
                    print_cnf_clauses=True,
                )
        if u_is_binary:
            _print_clause_reassembly_from_u_vector(
                u_vector=optimized.u_vector,
                T=T,
                V=V,
                start=start,
                goal=goal,
                domain_edges=domain_edges,
                bob_edges_by_party=bob_payloads,
            )
        else:
            print("=== Clause reassembly from u_vector ===")
            print("Skipped because u_vector is non-binary.")

    # Assertions happen last so all debug info above is printed on failure.
    if has_u:
        assert u_is_binary, "Expected binary u_vector but got non-binary values"
    assert bool(optimized.is_solved) is bool(expect_sat)
    assert optimized.information_gain == 0.0

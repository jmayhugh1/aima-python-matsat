import pytest
import time
from logic4e import dpll_satisfiable
from private_path_query_logic import (
    physics,
    bob_physics,
    alice_physics,
    PosBit,
    num_bits,
    Active,
    Allowed,
    directed_pairs_from_edges,
    ordered_symbols,
    assignment_from_u_vector,
    decode_path_from_assignment,
    build_q,
    build_physics_q,
    build_bob_q,
    build_alice_q,
    # Internal helpers for setup calculations (row counts)
    _build_alice_q,
    _build_bob_q,
)
from private_path_query_utils import (
    Graph,
    GraphPath,
    Vertex,
    Edge,
    EdgeState,
    PrivatePathInfo,
    compile_find_safe_path,
)
from tests.private_path_query_test_helpers import (
    _and_all,
    _force_graph_path_moves,
    _print_sat_moves,
    _print_assignments_from_u_vector,
    _assert_optimized_result,
    _unknown_domain_edges_from_graph,
    _run_find_safe_path_compare_modes,
    _make_private_path_info,
    _verify_result_and_print,
)


def _make_info(
    T: int,
    V: int,
    edges: list[Edge] | None = None,
    use_edge_domain: bool = False,
) -> PrivatePathInfo:
    """
    Helper to create PrivatePathInfo for tests.
    Creates minimal valid info with 2 parties (Alice + 1 Bob).
    """
    edge_domain_edges = None
    if use_edge_domain and edges:
        # Convert edges to UNKNOWN state for domain
        edge_domain_edges = [
            Edge(e.vertex1, e.vertex2, EdgeState.UNKNOWN) for e in edges
        ]
    return PrivatePathInfo(
        num_parties=2,
        T=T,
        V=V,
        rows_per_id=[1, 1],  # Placeholder - actual rows computed by build functions
        use_edge_domain=use_edge_domain,
        edge_domain_edges=edge_domain_edges,
    )


def test_private_path_info_from_dict():
    info = PrivatePathInfo.from_dict(
        {
            "num_parties": 3,
            "T": 2,
            "V": 4,
            "rows_per_id": [10, 20, 20],
            "use_weight_vector": True,
            "use_edge_domain": True,
            "edge_domain_pairs": [[0, 1], [1, 2]],
        }
    )
    assert info.num_parties == 3
    assert info.T == 2
    assert info.V == 4
    assert info.rows_per_id == [10, 20, 20]
    assert info.use_weight_vector
    assert info.use_edge_domain
    assert info.edge_domain_edges is not None
    assert all(e.state == EdgeState.UNKNOWN for e in info.edge_domain_edges)


def test_private_path_info_from_json_file(tmp_path):
    config_path = tmp_path / "private_path_info.json"
    config_path.write_text(
        '{"num_parties": 3, "T": 1, "V": 2, "rows_per_id": [2, 15, 15], "use_edge_domain": true, "edge_domain_pairs": [[0,1]]}',
        encoding="utf-8",
    )
    info = PrivatePathInfo.from_json_file(str(config_path))
    assert info.num_parties == 3
    assert info.T == 1
    assert info.V == 2
    assert info.rows_per_id == [2, 15, 15]
    assert not info.use_weight_vector
    assert info.use_edge_domain
    assert info.edge_domain_edges is not None
    assert len(info.edge_domain_edges) == 1
    assert info.edge_domain_edges[0].state == EdgeState.UNKNOWN


def test_private_path_info_invalid_rows_length():
    with pytest.raises(ValueError, match="rows_per_id length must match num_parties"):
        PrivatePathInfo(num_parties=3, T=1, V=2, rows_per_id=[2, 15])


def test_private_path_info_edge_domain_requires_unknown_edges():
    with pytest.raises(ValueError, match="UNKNOWN"):
        PrivatePathInfo(
            num_parties=3,
            T=1,
            V=3,
            rows_per_id=[2, 10, 10],
            use_edge_domain=True,
            edge_domain_edges=[Edge(Vertex(0), Vertex(1), EdgeState.TRAVERSABLE)],
        )


@pytest.mark.asyncio
async def test_compile_find_safe_path_with_private_path_info_real_compile():
    T, V = 1, 2
    syms = ordered_symbols(T, V)
    q_alice, _ = _build_alice_q(start=0, goal=0, T=T, V=V, symbols=syms)
    q_bob, _ = _build_bob_q(edges=[], T=T, V=V, symbols=syms)
    q_physics, _ = build_physics_q(T=T, V=V, symbols=syms)
    alice_rows = int(q_physics.shape[0] + q_alice.shape[0])
    bob_rows = int(q_bob.shape[0])
    info = PrivatePathInfo(
        num_parties=3,
        T=T,
        V=V,
        rows_per_id=[alice_rows, bob_rows, bob_rows],
    )
    ok = await compile_find_safe_path(private_path_info=info)
    assert ok


@pytest.mark.asyncio
async def test_compile_find_safe_path_with_public_edge_domain_real_compile():
    T, V = 2, 3
    public_domain_edges = [
        Edge(Vertex(0), Vertex(1), EdgeState.UNKNOWN),
        Edge(Vertex(1), Vertex(2), EdgeState.UNKNOWN),
    ]
    compact_pairs = directed_pairs_from_edges(public_domain_edges, V)
    syms = ordered_symbols(T, V, directed_pairs=compact_pairs)
    q_alice, _ = _build_alice_q(start=0, goal=2, T=T, V=V, symbols=syms)
    q_bob, _ = _build_bob_q(
        edges=[], T=T, V=V, symbols=syms, directed_pairs=compact_pairs
    )
    q_physics, _ = build_physics_q(T=T, V=V, symbols=syms, directed_pairs=compact_pairs)
    alice_rows = int(q_physics.shape[0] + q_alice.shape[0])
    bob_rows = int(q_bob.shape[0])
    info = PrivatePathInfo(
        num_parties=3,
        T=T,
        V=V,
        rows_per_id=[alice_rows, bob_rows, bob_rows],
        use_edge_domain=True,
        edge_domain_edges=public_domain_edges,
    )
    ok = await compile_find_safe_path(private_path_info=info)
    assert ok


def test_logic_graphpath_sat_traversable_edges():
    v0, v1, v2 = Vertex(0), Vertex(1), Vertex(2)
    e01 = Edge(v0, v1, EdgeState.TRAVERSABLE)
    e12 = Edge(v1, v2, EdgeState.TRAVERSABLE)

    graph = Graph(vertices=[v0, v1, v2], edges=[e01, e12])
    path = GraphPath(start=v0, moves=[Edge(v0, v1), Edge(v1, v2)])

    T, V = len(path.moves), len(graph.vertices)
    clauses = []
    clauses.extend(physics(T=T, V=V))
    clauses.extend(bob_physics(edges=graph.edges, V=V)[0])
    clauses.extend(alice_physics(start=path.start.id, goal=path.end.id, T=T, V=V))
    clauses.extend(_force_graph_path_moves(path))

    model = dpll_satisfiable(_and_all(clauses))
    assert model
    _print_sat_moves(model, T, V)


def test_logic_graphpath_unsat_blocked_edge():
    v0, v1, v2 = Vertex(0), Vertex(1), Vertex(2)
    e01 = Edge(v0, v1, EdgeState.TRAVERSABLE)
    e12_blocked = Edge(v1, v2, EdgeState.BLOCKED)

    graph = Graph(vertices=[v0, v1, v2], edges=[e01, e12_blocked])
    path = GraphPath(start=v0, moves=[Edge(v0, v1), Edge(v1, v2)])

    T, V = len(path.moves), len(graph.vertices)
    clauses = []
    clauses.extend(physics(T=T, V=V))
    clauses.extend(bob_physics(edges=graph.edges, V=V)[0])
    clauses.extend(alice_physics(start=path.start.id, goal=path.end.id, T=T, V=V))
    clauses.extend(_force_graph_path_moves(path))

    assert not dpll_satisfiable(_and_all(clauses))


def test_logic_graphpath_unsat_no_edge():
    v0, v1, v2 = Vertex(0), Vertex(1), Vertex(2)
    e01 = Edge(v0, v1, EdgeState.TRAVERSABLE)

    graph = Graph(vertices=[v0, v1, v2], edges=[e01])
    path = GraphPath(start=v0, moves=[Edge(v0, v1), Edge(v1, v2)])

    T, V = len(path.moves), len(graph.vertices)
    clauses = []
    clauses.extend(physics(T=T, V=V))
    clauses.extend(bob_physics(edges=graph.edges, V=V)[0])
    clauses.extend(alice_physics(start=path.start.id, goal=path.end.id, T=T, V=V))
    clauses.extend(_force_graph_path_moves(path))

    assert not dpll_satisfiable(_and_all(clauses))


def test_logic_graphpath_sat_with_wait():
    v0, v1 = Vertex(0), Vertex(1)
    e01 = Edge(v0, v1, EdgeState.TRAVERSABLE)
    graph = Graph(vertices=[v0, v1], edges=[e01])
    path = GraphPath(start=v0, moves=[Edge(v0, v1)])

    # Allow one extra timestep; solver should use Wait at goal.
    T, V = 2, len(graph.vertices)
    clauses = []
    clauses.extend(physics(T=T, V=V))
    clauses.extend(bob_physics(edges=graph.edges, V=V)[0])
    clauses.extend(alice_physics(start=path.start.id, goal=path.end.id, T=T, V=V))
    clauses.extend(_force_graph_path_moves(path))  # constrain only t=0 move

    model = dpll_satisfiable(_and_all(clauses))
    assert model
    _print_sat_moves(model, T, V)


def test_logic_large_graph_sat_mixed_edges_forced_path_with_waits():
    """Larger SAT case with traversable/blocked/no-edge mix and extra timesteps."""
    vertices = [Vertex(i) for i in range(6)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[3], EdgeState.TRAVERSABLE),
        Edge(vertices[3], vertices[4], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
        Edge(vertices[2], vertices[5], EdgeState.BLOCKED),
    ]
    graph = Graph(vertices=vertices, edges=edges)
    path = GraphPath(
        start=vertices[0],
        moves=[
            Edge(vertices[0], vertices[1]),
            Edge(vertices[1], vertices[3]),
            Edge(vertices[3], vertices[4]),
        ],
    )

    # More timesteps than path length; model should use wait(s) after reaching goal.
    T, V = 5, len(vertices)
    clauses = []
    clauses.extend(physics(T=T, V=V))
    clauses.extend(bob_physics(edges=graph.edges, V=V)[0])
    clauses.extend(alice_physics(start=path.start.id, goal=path.end.id, T=T, V=V))
    clauses.extend(_force_graph_path_moves(path))

    model = dpll_satisfiable(_and_all(clauses))
    assert model
    _print_sat_moves(model, T, V)


def test_logic_large_graph_unsat_forced_blocked_transition():
    """Larger UNSAT case where forced route includes a blocked edge."""
    vertices = [Vertex(i) for i in range(5)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
        Edge(vertices[2], vertices[4], EdgeState.TRAVERSABLE),
    ]
    graph = Graph(vertices=vertices, edges=edges)
    path = GraphPath(
        start=vertices[0],
        moves=[
            Edge(vertices[0], vertices[1]),
            Edge(vertices[1], vertices[2]),
            Edge(vertices[2], vertices[4]),
        ],
    )

    T, V = len(path.moves), len(vertices)
    clauses = []
    clauses.extend(physics(T=T, V=V))
    clauses.extend(bob_physics(edges=graph.edges, V=V)[0])
    clauses.extend(alice_physics(start=path.start.id, goal=path.end.id, T=T, V=V))
    clauses.extend(_force_graph_path_moves(path))

    assert not dpll_satisfiable(_and_all(clauses))


def test_logic_large_graph_unsat_disconnected_goal():
    """Larger UNSAT case without forced moves: goal is in disconnected component."""
    vertices = [Vertex(i) for i in range(6)]
    # Component A: 0-1-2, Component B: 3-4-5, no bridge between components.
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.TRAVERSABLE),
        Edge(vertices[3], vertices[4], EdgeState.TRAVERSABLE),
        Edge(vertices[4], vertices[5], EdgeState.TRAVERSABLE),
    ]
    graph = Graph(vertices=vertices, edges=edges)

    T, V = 4, len(vertices)
    clauses = []
    clauses.extend(physics(T=T, V=V))
    clauses.extend(bob_physics(edges=graph.edges, V=V)[0])
    clauses.extend(alice_physics(start=0, goal=5, T=T, V=V))

    assert not dpll_satisfiable(_and_all(clauses))


@pytest.mark.asyncio
async def test_find_safe_path_fat_graph_mixed_states_sat_and_unsat():
    """
    Fat MPC test using real MP-SPDZ execution on a larger mixed-state graph.

    Runs both:
    - SAT case: traversable backbone path to goal exists
    - UNSAT case: one critical backbone edge changed to blocked
    """
    vertices = [Vertex(i) for i in range(8)]
    base_edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[3], EdgeState.TRAVERSABLE),
        Edge(vertices[3], vertices[5], EdgeState.TRAVERSABLE),
        Edge(vertices[5], vertices[7], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
        Edge(vertices[2], vertices[4], EdgeState.BLOCKED),
        # many omitted pairs => NO_EDGE
    ]

    # SAT graph pair
    sat_graphs = [
        Graph(vertices=vertices, edges=base_edges),
        Graph(vertices=vertices, edges=base_edges),
    ]
    sat_domain = _unknown_domain_edges_from_graph(sat_graphs[0])
    sat_info = _make_private_path_info(
        T=6,
        V=8,
        num_bobs=len(sat_graphs),
        use_edge_domain=True,
        edge_domain_edges=sat_domain,
    )
    sat_baseline, sat_optimized = await _run_find_safe_path_compare_modes(
        info=sat_info,
        graphs=sat_graphs,
        start=vertices[0],
        goal=vertices[7],
        port=5005,
    )
    if sat_baseline is not None:
        assert sat_baseline.is_solved
        assert sat_baseline.information_gain == 0.0
    if sat_optimized is not None:
        assert sat_optimized.is_solved
        assert sat_optimized.information_gain == 0.0

    # UNSAT graph pair: break the only backbone route to goal.
    unsat_edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[3], EdgeState.TRAVERSABLE),
        Edge(vertices[3], vertices[5], EdgeState.BLOCKED),  # critical break
        Edge(vertices[5], vertices[7], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
        Edge(vertices[2], vertices[4], EdgeState.BLOCKED),
    ]
    unsat_graphs = [
        Graph(vertices=vertices, edges=unsat_edges),
        Graph(vertices=vertices, edges=unsat_edges),
    ]
    unsat_domain = _unknown_domain_edges_from_graph(unsat_graphs[0])
    unsat_info = _make_private_path_info(
        T=6,
        V=8,
        num_bobs=len(unsat_graphs),
        use_edge_domain=True,
        edge_domain_edges=unsat_domain,
    )
    unsat_baseline, unsat_optimized = await _run_find_safe_path_compare_modes(
        info=unsat_info,
        graphs=unsat_graphs,
        start=vertices[0],
        goal=vertices[7],
        port=5007,
    )
    if unsat_baseline is not None:
        assert not unsat_baseline.is_solved
        assert unsat_baseline.information_gain == 0.0
    if unsat_optimized is not None:
        assert not unsat_optimized.is_solved
        assert unsat_optimized.information_gain == 0.0


def test_ordered_symbols_is_deterministic():
    T, V = 2, 3
    syms_1 = ordered_symbols(T, V)
    syms_2 = ordered_symbols(T, V)
    assert syms_1 == syms_2

    # Spot-check expected prefix/suffix ordering by category.
    assert syms_1[0] == PosBit(0, 0)
    assert syms_1[1] == PosBit(0, 1)
    assert syms_1[2] == PosBit(1, 0)
    assert syms_1[-1] == Allowed(2, 1)


def test_ordered_symbols_exact_small_case():
    """
    Golden test: exact symbol order for T=1, V=2.
    This is intentionally explicit so readers can trust the encoding contract.
    """
    syms = ordered_symbols(T=1, V=2)
    got = [str(s) for s in syms]
    expected = [
        "PosBit(0, 0)",
        "PosBit(1, 0)",
        "Active(0, 1)",
        "Active(1, 0)",
        "Allowed(0, 1)",
        "Allowed(1, 0)",
    ]
    assert got == expected


def test_assignment_from_u_vector_exact_small_case():
    T, V = 1, 2
    syms = ordered_symbols(T, V)
    # Matches the golden symbol order exactly.
    u_vector = [0, 1, 1, 0, 1, 0]
    assignment = assignment_from_u_vector(u_vector=u_vector, T=T, V=V, symbols=syms)

    assert assignment[PosBit(0, 0)] is False
    assert assignment[PosBit(1, 0)] is True
    assert assignment[Active(0, 1)] is True
    assert assignment[Active(1, 0)] is False
    assert assignment[Allowed(0, 1)] is True
    assert assignment[Allowed(1, 0)] is False
    assert decode_path_from_assignment(assignment, T=T, V=V) == [0, 1]


def test_build_q_block_shapes_and_vertical_concat():
    vertices = [Vertex(i) for i in range(4)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
    ]
    T, V = 2, 4
    info = _make_info(T=T, V=V, edges=edges)
    q_full, syms, blocks = build_q(info, start=0, goal=2, edges=edges)

    assert q_full.shape[1] == 2 * len(syms)
    assert blocks["physics"].shape[1] == q_full.shape[1]
    assert blocks["bob"].shape[1] == q_full.shape[1]
    assert blocks["alice"].shape[1] == q_full.shape[1]
    assert q_full.shape[0] == (
        blocks["physics"].shape[0] + blocks["bob"].shape[0] + blocks["alice"].shape[0]
    )
    # Vertical concatenation contract: q_full is exactly [physics; bob; alice].
    # This assertion avoids accidental reorder/regression of block assembly.
    assert (q_full[: blocks["physics"].shape[0]] == blocks["physics"]).all()
    bob_start = blocks["physics"].shape[0]
    bob_end = bob_start + blocks["bob"].shape[0]
    assert (q_full[bob_start:bob_end] == blocks["bob"]).all()
    assert (q_full[bob_end:] == blocks["alice"]).all()


def test_build_q_alice_unit_clauses_present():
    """Alice block should encode start/goal bit-unit clauses."""
    vertices = [Vertex(i) for i in range(3)]
    edges = [Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE)]
    T, V = 2, 3
    start, goal = 0, 2
    info = _make_info(T=T, V=V, edges=edges)
    _, syms, blocks = build_q(info, start=start, goal=goal, edges=edges)

    n = len(syms)
    b = num_bits(V)
    alice = blocks["alice"]

    for k in range(b):
        start_bit = (start >> k) & 1
        goal_bit = (goal >> k) & 1
        idx_start_bit = syms.index(PosBit(0, k))
        idx_goal_bit = syms.index(PosBit(T, k))
        if start_bit == 1:
            assert (alice[:, idx_start_bit] == 1).any()
        else:
            assert (alice[:, n + idx_start_bit] == 1).any()
        if goal_bit == 1:
            assert (alice[:, idx_goal_bit] == 1).any()
        else:
            assert (alice[:, n + idx_goal_bit] == 1).any()


def test_build_q_bob_encodes_edge_state_semantics():
    """Bob block should include clauses for Active/Allowed based on edge states."""
    vertices = [Vertex(i) for i in range(3)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),  # Active + Allowed
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),  # Active + ~Allowed
    ]
    T, V = 1, 3
    info = _make_info(T=T, V=V, edges=edges)
    _, syms, blocks = build_q(info, start=0, goal=2, edges=edges)
    bob = blocks["bob"]
    n = len(syms)

    i_active_01 = syms.index(Active(0, 1))
    i_allowed_01 = syms.index(Allowed(0, 1))
    i_active_12 = syms.index(Active(1, 2))
    i_allowed_12 = syms.index(Allowed(1, 2))

    # Traversable edge: should assert positive Active and positive Allowed somewhere.
    assert (bob[:, i_active_01] == 1).any()
    assert (bob[:, i_allowed_01] == 1).any()

    # Blocked edge: should assert positive Active and negated Allowed somewhere.
    assert (bob[:, i_active_12] == 1).any()
    assert (bob[:, n + i_allowed_12] == 1).any()


def test_build_physics_q_standalone_matches_build_q_block():
    T, V = 2, 4
    vertices = [Vertex(i) for i in range(V)]
    edges = [Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE)]
    syms = ordered_symbols(T, V)
    info = _make_info(T=T, V=V, edges=edges)

    q_physics, _ = build_physics_q(T=T, V=V, symbols=syms)
    q_full, syms_full, blocks = build_q(info, start=0, goal=1, edges=edges)

    assert syms == syms_full
    assert q_physics.shape[1] == 2 * len(syms)
    assert q_physics.shape[0] > 0
    assert (q_physics == blocks["physics"]).all()
    assert (q_full[: q_physics.shape[0]] == q_physics).all()


def test_build_bob_q_standalone_matches_build_q_block():
    T, V = 1, 3
    vertices = [Vertex(i) for i in range(V)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
    ]
    info = _make_info(T=T, V=V, edges=edges)

    q_bob, _ = build_bob_q(info, edges=edges)
    _, syms_full, blocks = build_q(info, start=0, goal=2, edges=edges)

    assert q_bob.shape[1] == 2 * len(syms_full)
    assert q_bob.shape[0] > 0
    assert (q_bob == blocks["bob"]).all()


def test_build_alice_q_standalone_matches_build_q_block():
    T, V = 3, 5
    start, goal = 1, 4
    vertices = [Vertex(i) for i in range(V)]
    edges = [Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE)]
    info = _make_info(T=T, V=V, edges=edges)

    q_alice, _ = build_alice_q(info, start=start, goal=goal)
    _, syms_full, blocks = build_q(info, start=start, goal=goal, edges=edges)

    assert q_alice.shape[1] == 2 * len(syms_full)
    assert q_alice.shape[0] > 0
    assert (q_alice == blocks["alice"]).all()


def test_build_q_edge_domain_flag_reduces_size_and_runtime():
    vertices = [Vertex(i) for i in range(8)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.TRAVERSABLE),
        Edge(vertices[2], vertices[3], EdgeState.BLOCKED),
        Edge(vertices[3], vertices[4], EdgeState.TRAVERSABLE),
    ]
    T, V = 4, len(vertices)

    # Full (no edge domain)
    info_full = _make_info(T=T, V=V, edges=edges, use_edge_domain=False)
    t0 = time.perf_counter()
    q_full, syms_full, _ = build_q(info_full, start=0, goal=4, edges=edges)
    full_elapsed = time.perf_counter() - t0

    # Compact (with edge domain)
    info_compact = _make_info(T=T, V=V, edges=edges, use_edge_domain=True)
    t1 = time.perf_counter()
    q_compact, syms_compact, _ = build_q(info_compact, start=0, goal=4, edges=edges)
    compact_elapsed = time.perf_counter() - t1

    print(
        f"build_q runtime baseline={full_elapsed:.4f}s compact={compact_elapsed:.4f}s "
        f"cols {q_full.shape[1]}->{q_compact.shape[1]} rows {q_full.shape[0]}->{q_compact.shape[0]}"
    )
    assert q_compact.shape[1] < q_full.shape[1]
    assert q_compact.shape[0] < q_full.shape[0]
    assert len(syms_compact) < len(syms_full)


def test_build_physics_q_edge_domain_runtime_compare():
    vertices = [Vertex(i) for i in range(10)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.TRAVERSABLE),
        Edge(vertices[2], vertices[5], EdgeState.TRAVERSABLE),
        Edge(vertices[5], vertices[7], EdgeState.BLOCKED),
    ]
    T, V = 4, len(vertices)
    compact_pairs = directed_pairs_from_edges(edges, V)

    t0 = time.perf_counter()
    q_full, _ = build_physics_q(T=T, V=V)
    full_elapsed = time.perf_counter() - t0

    syms_compact = ordered_symbols(T=T, V=V, directed_pairs=compact_pairs)
    t1 = time.perf_counter()
    q_compact, _ = build_physics_q(
        T=T, V=V, symbols=syms_compact, directed_pairs=compact_pairs
    )
    compact_elapsed = time.perf_counter() - t1

    print(
        f"build_physics_q runtime baseline={full_elapsed:.4f}s compact={compact_elapsed:.4f}s "
        f"cols {q_full.shape[1]}->{q_compact.shape[1]} rows {q_full.shape[0]}->{q_compact.shape[0]}"
    )
    assert q_compact.shape[1] < q_full.shape[1]
    assert q_compact.shape[0] < q_full.shape[0]


@pytest.mark.asyncio
async def test_find_safe_path_real_example_sat():
    v0, v1 = Vertex(id=0), Vertex(id=1)
    vertices = [v0, v1]
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE)]
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    public_domain_edges = [Edge(Vertex(0), Vertex(1), EdgeState.UNKNOWN)]
    info = _make_private_path_info(
        T=1,
        V=2,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=public_domain_edges,
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(1),
        port=5003,
    )
    baseline_syms = ordered_symbols(T=info.T, V=info.V)
    optimized_syms = ordered_symbols(
        T=1, V=2, directed_pairs=directed_pairs_from_edges(public_domain_edges, 2)
    )
    print(
        f"baseline is solved: {None if baseline is None else baseline.is_solved}, "
        f"optimized is solved: {None if optimized is None else optimized.is_solved}"
    )
    _verify_result_and_print(baseline, expect_sat=True, T=1, V=2, symbols=baseline_syms)
    _verify_result_and_print(
        optimized,
        expect_sat=True,
        T=1,
        V=2,
        symbols=optimized_syms,
        start=0,
        goal=1,
        domain_edges=public_domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


@pytest.mark.asyncio
async def test_find_safe_path_real_example_sat_nine_vertices():
    vertices = [Vertex(id=i) for i in range(5)]
    # Keep a few mixed-state edges, but make SAT trivial with T=0 and start==goal.
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[4], EdgeState.TRAVERSABLE),
        Edge(vertices[2], vertices[3], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
        Edge(vertices[3], vertices[4], EdgeState.TRAVERSABLE),
    ]
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    default_domain = _unknown_domain_edges_from_graph(graphs[0])
    info = _make_private_path_info(
        T=5,
        V=5,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=default_domain,
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(4),
        port=5010,
    )
    baseline_syms = ordered_symbols(T=info.T, V=info.V)
    optimized_syms = ordered_symbols(
        T=5, V=5, directed_pairs=directed_pairs_from_edges(default_domain, 5)
    )
    _verify_result_and_print(baseline, expect_sat=True, T=5, V=5, symbols=baseline_syms)
    _verify_result_and_print(
        optimized,
        expect_sat=True,
        T=5,
        V=5,
        symbols=optimized_syms,
        start=0,
        goal=4,
        domain_edges=default_domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


@pytest.mark.asyncio
async def test_find_safe_path_real_example_unsat():
    v0, v1 = Vertex(id=0), Vertex(id=1)
    vertices = [v0, v1]
    edges = [Edge(v0, v1, EdgeState.BLOCKED)]
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]

    public_domain_edges = [Edge(Vertex(0), Vertex(1), EdgeState.UNKNOWN)]
    info = _make_private_path_info(
        T=1,
        V=2,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=public_domain_edges,
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(1),
        port=5004,
    )
    baseline_syms = ordered_symbols(T=info.T, V=info.V)
    optimized_syms = ordered_symbols(
        T=info.T,
        V=info.V,
        directed_pairs=directed_pairs_from_edges(public_domain_edges, info.V),
    )
    _verify_result_and_print(
        baseline, expect_sat=False, T=1, V=2, symbols=baseline_syms
    )
    _verify_result_and_print(
        optimized,
        expect_sat=False,
        T=1,
        V=2,
        symbols=optimized_syms,
        start=0,
        goal=1,
        domain_edges=public_domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


def _assert_exact_u_vector_order_single_edge_case(
    *,
    optimized,
    domain: list[Edge],
    expect_allowed: bool,
):
    assert optimized is not None
    assert optimized.u_vector is not None
    compact_pairs = directed_pairs_from_edges(domain, 2)
    assert compact_pairs == [(0, 1)]
    # Compact symbol order for T=1, V=2 and one directed pair.
    expected_symbol_order = [
        "PosBit(0, 0)",
        "PosBit(1, 0)",
        "Active(0, 1)",
        "Allowed(0, 1)",
    ]
    got_symbol_order = [
        str(s) for s in ordered_symbols(T=1, V=2, directed_pairs=compact_pairs)
    ]
    assert got_symbol_order == expected_symbol_order

    expected_u = [0, 1, 1, 1 if expect_allowed else 0]
    assert (
        optimized.u_vector == expected_u
    ), f"Expected exact u-vector ordering {expected_u}, got {optimized.u_vector}"


@pytest.mark.asyncio
async def test_optimized_matsat_sat_single_traversable_edge():
    v0, v1 = Vertex(0), Vertex(1)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE)]
    graphs = [Graph([v0, v1], edges=edges), Graph([v0, v1], edges=edges)]
    domain = [Edge(v0, v1, EdgeState.UNKNOWN)]
    info = _make_private_path_info(
        T=1, V=2, num_bobs=len(graphs), use_edge_domain=True, edge_domain_edges=domain
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=v0,
        goal=v1,
        port=5100,
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=v0.id,
        goal=v1.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )
    _assert_exact_u_vector_order_single_edge_case(
        optimized=optimized, domain=domain, expect_allowed=True
    )


@pytest.mark.asyncio
async def test_optimized_matsat_unsat_single_blocked_edge():
    v0, v1 = Vertex(0), Vertex(1)
    edges = [Edge(v0, v1, EdgeState.BLOCKED)]
    graphs = [Graph([v0, v1], edges=edges), Graph([v0, v1], edges=edges)]
    domain = [Edge(v0, v1, EdgeState.UNKNOWN)]
    info = _make_private_path_info(
        T=1, V=2, num_bobs=len(graphs), use_edge_domain=True, edge_domain_edges=domain
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=v0,
        goal=v1,
        port=5102,
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=False,
        T=info.T,
        V=info.V,
        start=v0.id,
        goal=v1.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
        print_cnf=True,
    )


@pytest.mark.asyncio
async def test_optimized_matsat_sat_chain_two_steps():
    v0, v1, v2 = Vertex(0), Vertex(1), Vertex(2)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE), Edge(v1, v2, EdgeState.TRAVERSABLE)]
    graphs = [Graph([v0, v1, v2], edges=edges), Graph([v0, v1, v2], edges=edges)]
    domain = _unknown_domain_edges_from_graph(graphs[0])
    info = _make_private_path_info(
        T=2, V=3, num_bobs=len(graphs), use_edge_domain=True, edge_domain_edges=domain
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=v0,
        goal=v2,
        port=5106,
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=v0.id,
        goal=v2.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


@pytest.mark.asyncio
async def test_optimized_matsat_unsat_insufficient_horizon():
    v0, v1, v2 = Vertex(0), Vertex(1), Vertex(2)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE), Edge(v1, v2, EdgeState.TRAVERSABLE)]
    graphs = [Graph([v0, v1, v2], edges=edges), Graph([v0, v1, v2], edges=edges)]
    domain = _unknown_domain_edges_from_graph(graphs[0])
    info = _make_private_path_info(
        T=1, V=3, num_bobs=len(graphs), use_edge_domain=True, edge_domain_edges=domain
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=v0,
        goal=v2,
        port=5108,
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=False,
        T=info.T,
        V=info.V,
        start=v0.id,
        goal=v2.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


@pytest.mark.asyncio
async def test_optimized_matsat_sat_detour_around_blocked_edge():
    v0, v1, v2, v3 = Vertex(0), Vertex(1), Vertex(2), Vertex(3)
    edges = [
        Edge(v0, v1, EdgeState.BLOCKED),
        Edge(v0, v2, EdgeState.TRAVERSABLE),
        Edge(v2, v3, EdgeState.TRAVERSABLE),
        Edge(v1, v3, EdgeState.TRAVERSABLE),
    ]
    graphs = [
        Graph([v0, v1, v2, v3], edges=edges),
        Graph([v0, v1, v2, v3], edges=edges),
    ]
    domain = _unknown_domain_edges_from_graph(graphs[0])
    info = _make_private_path_info(
        T=2, V=4, num_bobs=len(graphs), use_edge_domain=True, edge_domain_edges=domain
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=v0,
        goal=v3,
        port=5110,
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=v0.id,
        goal=v3.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
        print_cnf=True,
    )


@pytest.mark.asyncio
async def test_optimized_matsat_unsat_disconnected_goal():
    v0, v1, v2, v3 = Vertex(0), Vertex(1), Vertex(2), Vertex(3)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE)]
    graphs = [
        Graph([v0, v1, v2, v3], edges=edges),
        Graph([v0, v1, v2, v3], edges=edges),
    ]
    domain = [Edge(v0, v1, EdgeState.UNKNOWN), Edge(v1, v0, EdgeState.UNKNOWN)]
    info = _make_private_path_info(
        T=3, V=4, num_bobs=len(graphs), use_edge_domain=True, edge_domain_edges=domain
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=v0,
        goal=v3,
        port=5112,
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=False,
        T=info.T,
        V=info.V,
        start=v0.id,
        goal=v3.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
        print_cnf=True,
    )


@pytest.mark.asyncio
async def test_optimized_matsat_sat_reach_then_wait():
    v0, v1 = Vertex(0), Vertex(1)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE)]
    graphs = [Graph([v0, v1], edges=edges), Graph([v0, v1], edges=edges)]
    domain = [Edge(v0, v1, EdgeState.UNKNOWN), Edge(v1, v0, EdgeState.UNKNOWN)]
    info = _make_private_path_info(
        T=2, V=2, num_bobs=len(graphs), use_edge_domain=True, edge_domain_edges=domain
    )
    baseline, optimized = await _run_find_safe_path_compare_modes(
        info=info,
        graphs=graphs,
        start=v0,
        goal=v1,
        port=5114,
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=v0.id,
        goal=v1.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
        print_cnf=True,
    )

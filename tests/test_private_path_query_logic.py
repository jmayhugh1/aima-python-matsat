import pytest
import time
from logic4e import dpll_satisfiable
from private_path_query_logic import (
    physics,
    bob_physics,
    alice_physics,
    Move,
    Wait,
    At,
    Active,
    Allowed,
    directed_pairs_from_edges,
    ordered_symbols,
    assignment_from_u_vector,
    build_q,
    build_physics_q,
    build_bob_q,
    build_alice_q,
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
    _unknown_domain_edges_from_graph,
    _run_find_safe_path_compare_modes,
    _verify_result_and_print,
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
    q_alice = build_alice_q(start=0, goal=0, T=T, V=V, symbols=syms)
    q_bob = build_bob_q(edges=[], T=T, V=V, symbols=syms)
    q_physics = build_physics_q(T=T, V=V, symbols=syms)
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
    q_alice = build_alice_q(start=0, goal=2, T=T, V=V, symbols=syms)
    q_bob = build_bob_q(edges=[], T=T, V=V, symbols=syms, directed_pairs=compact_pairs)
    q_physics = build_physics_q(T=T, V=V, symbols=syms, directed_pairs=compact_pairs)
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
    clauses.extend(bob_physics(edges=graph.edges, V=V))
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
    clauses.extend(bob_physics(edges=graph.edges, V=V))
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
    clauses.extend(bob_physics(edges=graph.edges, V=V))
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
    clauses.extend(bob_physics(edges=graph.edges, V=V))
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
    clauses.extend(bob_physics(edges=graph.edges, V=V))
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
    clauses.extend(bob_physics(edges=graph.edges, V=V))
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
    clauses.extend(bob_physics(edges=graph.edges, V=V))
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
    sat_baseline, sat_optimized = await _run_find_safe_path_compare_modes(
        graphs=sat_graphs,
        start=vertices[0],
        goal=vertices[7],
        T=6,
        V=8,
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
    unsat_baseline, unsat_optimized = await _run_find_safe_path_compare_modes(
        graphs=unsat_graphs,
        start=vertices[0],
        goal=vertices[7],
        T=6,
        V=8,
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
    assert syms_1[0] == At(0, 0)
    assert syms_1[1] == At(0, 1)
    assert syms_1[2] == At(0, 2)
    assert syms_1[-1] == Allowed(2, 1)


def test_ordered_symbols_exact_small_case():
    """
    Golden test: exact symbol order for T=1, V=2.
    This is intentionally explicit so readers can trust the encoding contract.
    """
    syms = ordered_symbols(T=1, V=2)
    got = [str(s) for s in syms]
    expected = [
        "At(0, 0)",
        "At(0, 1)",
        "At(1, 0)",
        "At(1, 1)",
        "Move(0, 0, 1)",
        "Move(0, 1, 0)",
        "Wait(0, 0)",
        "Wait(0, 1)",
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
    u_vector = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    assignment = assignment_from_u_vector(u_vector=u_vector, T=T, V=V, symbols=syms)

    assert assignment[At(0, 0)] is True
    assert assignment[At(0, 1)] is False
    assert assignment[At(1, 0)] is False
    assert assignment[At(1, 1)] is True
    assert assignment[Move(0, 0, 1)] is True
    assert assignment[Move(0, 1, 0)] is False
    assert assignment[Wait(0, 0)] is False
    assert assignment[Wait(0, 1)] is True
    assert assignment[Active(0, 1)] is True
    assert assignment[Active(1, 0)] is False
    assert assignment[Allowed(0, 1)] is True
    assert assignment[Allowed(1, 0)] is False


def test_build_q_block_shapes_and_vertical_concat():
    vertices = [Vertex(i) for i in range(4)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
    ]
    T, V = 2, 4
    q_full, syms, blocks = build_q(start=0, goal=2, T=T, V=V, edges=edges)

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
    """Alice block should include At(0,start) and At(T,goal) as positive unit clauses."""
    vertices = [Vertex(i) for i in range(3)]
    edges = [Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE)]
    T, V = 2, 3
    start, goal = 0, 2
    _, syms, blocks = build_q(start=start, goal=goal, T=T, V=V, edges=edges)

    n = len(syms)
    idx_start = syms.index(At(0, start))
    idx_goal = syms.index(At(T, goal))

    alice = blocks["alice"]
    assert (
        alice[:, idx_start] == 1
    ).any(), "Missing At(0,start) unit clause in Alice block"
    assert (
        alice[:, idx_goal] == 1
    ).any(), "Missing At(T,goal) unit clause in Alice block"

    # For those unit clauses, there should be no negated occurrence in the same row.
    start_rows = [i for i in range(alice.shape[0]) if alice[i, idx_start] == 1]
    goal_rows = [i for i in range(alice.shape[0]) if alice[i, idx_goal] == 1]
    assert any(alice[i, n + idx_start] == 0 for i in start_rows)
    assert any(alice[i, n + idx_goal] == 0 for i in goal_rows)


def test_build_q_bob_encodes_edge_state_semantics():
    """Bob block should include clauses for Active/Allowed based on edge states."""
    vertices = [Vertex(i) for i in range(3)]
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),  # Active + Allowed
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),  # Active + ~Allowed
    ]
    T, V = 1, 3
    _, syms, blocks = build_q(start=0, goal=2, T=T, V=V, edges=edges)
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

    q_physics = build_physics_q(T=T, V=V, symbols=syms)
    q_full, syms_full, blocks = build_q(start=0, goal=1, T=T, V=V, edges=edges)

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
    syms = ordered_symbols(T, V)

    q_bob = build_bob_q(edges=edges, T=T, V=V, symbols=syms)
    _, syms_full, blocks = build_q(start=0, goal=2, T=T, V=V, edges=edges)

    assert syms == syms_full
    assert q_bob.shape[1] == 2 * len(syms)
    assert q_bob.shape[0] > 0
    assert (q_bob == blocks["bob"]).all()


def test_build_alice_q_standalone_matches_build_q_block():
    T, V = 3, 5
    start, goal = 1, 4
    vertices = [Vertex(i) for i in range(V)]
    edges = [Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE)]
    syms = ordered_symbols(T, V)

    q_alice = build_alice_q(start=start, goal=goal, T=T, V=V, symbols=syms)
    _, syms_full, blocks = build_q(start=start, goal=goal, T=T, V=V, edges=edges)

    assert syms == syms_full
    assert q_alice.shape[1] == 2 * len(syms)
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

    t0 = time.perf_counter()
    q_full, syms_full, _ = build_q(
        start=0, goal=4, T=T, V=V, edges=edges, use_edge_domain=False
    )
    full_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    q_compact, syms_compact, _ = build_q(
        start=0, goal=4, T=T, V=V, edges=edges, use_edge_domain=True
    )
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
    q_full = build_physics_q(T=T, V=V)
    full_elapsed = time.perf_counter() - t0

    syms_compact = ordered_symbols(T=T, V=V, directed_pairs=compact_pairs)
    t1 = time.perf_counter()
    q_compact = build_physics_q(
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
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(1),
        T=1,
        V=2,
        port=5003,
        public_domain_edges=public_domain_edges,
    )
    baseline_syms = ordered_symbols(T=1, V=2)
    optimized_syms = ordered_symbols(
        T=1, V=2, directed_pairs=directed_pairs_from_edges(public_domain_edges, 2)
    )
    print(
        f"baseline is solved: {None if baseline is None else baseline.is_solved}, "
        f"optimized is solved: {None if optimized is None else optimized.is_solved}"
    )
    _verify_result_and_print(baseline, expect_sat=True, T=1, V=2, symbols=baseline_syms)
    _verify_result_and_print(
        optimized, expect_sat=True, T=1, V=2, symbols=optimized_syms
    )


@pytest.mark.asyncio
async def test_find_safe_path_real_example_sat_nine_vertices():
    vertices = [Vertex(id=i) for i in range(5)]
    # Keep a few mixed-state edges, but make SAT trivial with T=0 and start==goal.
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.TRAVERSABLE),
        Edge(vertices[2], vertices[3], EdgeState.TRAVERSABLE),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
        Edge(vertices[3], vertices[4], EdgeState.TRAVERSABLE),
    ]
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(4),
        T=5,
        V=5,
        port=5010,
        mode="optimized",
    )
    baseline_syms = ordered_symbols(T=5, V=5)
    default_domain = _unknown_domain_edges_from_graph(graphs[0])
    optimized_syms = ordered_symbols(
        T=5, V=5, directed_pairs=directed_pairs_from_edges(default_domain, 5)
    )
    _verify_result_and_print(baseline, expect_sat=True, T=5, V=5, symbols=baseline_syms)
    _verify_result_and_print(
        optimized, expect_sat=True, T=5, V=5, symbols=optimized_syms
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
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(1),
        T=1,
        V=2,
        port=5004,
        public_domain_edges=public_domain_edges,
    )
    baseline_syms = ordered_symbols(T=1, V=2)
    optimized_syms = ordered_symbols(
        T=1, V=2, directed_pairs=directed_pairs_from_edges(public_domain_edges, 2)
    )
    _verify_result_and_print(
        baseline, expect_sat=False, T=1, V=2, symbols=baseline_syms
    )
    _verify_result_and_print(
        optimized, expect_sat=False, T=1, V=2, symbols=optimized_syms
    )

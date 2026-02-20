import pytest

from private_path_query_logic import directed_pairs_from_edges, ordered_symbols
from private_path_query_utils import (
    Graph,
    Vertex,
    Edge,
    EdgeState,
)
from tests.private_path_query_test_helpers import (
    _assert_optimized_result,
    _unknown_domain_edges_from_graph,
    _run_find_safe_path_compare_modes,
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
        "At(0, 0)",
        "At(0, 1)",
        "At(1, 0)",
        "At(1, 1)",
        "Move(0, 0, 1)",
        "Wait(0, 0)",
        "Wait(0, 1)",
        "Active(0, 1)",
        "Allowed(0, 1)",
    ]
    got_symbol_order = [
        str(s) for s in ordered_symbols(T=1, V=2, directed_pairs=compact_pairs)
    ]
    assert got_symbol_order == expected_symbol_order

    expected_u = [1, 0, 0, 1, 1, 0, 0, 1, 1 if expect_allowed else 0]
    assert (
        optimized.u_vector == expected_u
    ), f"Expected exact u-vector ordering {expected_u}, got {optimized.u_vector}"


@pytest.mark.asyncio
async def test_optimized_matsat_sat_single_traversable_edge():
    v0, v1 = Vertex(0), Vertex(1)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE)]
    graphs = [Graph([v0, v1], edges=edges), Graph([v0, v1], edges=edges)]
    domain = [Edge(v0, v1, EdgeState.UNKNOWN)]
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=v0,
        goal=v1,
        T=1,
        V=2,
        port=5100,
        public_domain_edges=domain,
        mode="optimized",
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=1,
        V=2,
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
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=v0,
        goal=v1,
        T=1,
        V=2,
        port=5102,
        public_domain_edges=domain,
        mode="optimized",
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=False,
        T=1,
        V=2,
        start=v0.id,
        goal=v1.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
        print_bob_cnf=True,
    )


# @pytest.mark.asyncio
# async def test_optimized_matsat_sat_wait_at_goal():
#     v0 = Vertex(0)
#     graphs = [Graph([v0], edges=[]), Graph([v0], edges=[])]
#     domain: list[Edge] = []
#     baseline, optimized = await _run_find_safe_path_compare_modes(
#         graphs=graphs,
#         start=v0,
#         goal=v0,
#         T=2,
#         V=1,
#         port=5104,
#         public_domain_edges=domain,
#         mode="optimized",
#     )
#     assert baseline is None
#     _assert_optimized_result(optimized, expect_sat=True, T=2, V=1, domain_edges=domain)


@pytest.mark.asyncio
async def test_optimized_matsat_sat_chain_two_steps():
    v0, v1, v2 = Vertex(0), Vertex(1), Vertex(2)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE), Edge(v1, v2, EdgeState.TRAVERSABLE)]
    graphs = [Graph([v0, v1, v2], edges=edges), Graph([v0, v1, v2], edges=edges)]
    domain = _unknown_domain_edges_from_graph(graphs[0])
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=v0,
        goal=v2,
        T=2,
        V=3,
        port=5106,
        public_domain_edges=domain,
        mode="optimized",
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=2,
        V=3,
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
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=v0,
        goal=v2,
        T=1,
        V=3,
        port=5108,
        public_domain_edges=domain,
        mode="optimized",
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=False,
        T=1,
        V=3,
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
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=v0,
        goal=v3,
        T=2,
        V=4,
        port=5110,
        public_domain_edges=domain,
        mode="optimized",
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=2,
        V=4,
        start=v0.id,
        goal=v3.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
        print_bob_cnf=True,
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
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=v0,
        goal=v3,
        T=3,
        V=4,
        port=5112,
        public_domain_edges=domain,
        mode="optimized",
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=False,
        T=3,
        V=4,
        start=v0.id,
        goal=v3.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


@pytest.mark.asyncio
async def test_optimized_matsat_sat_reach_then_wait():
    v0, v1 = Vertex(0), Vertex(1)
    edges = [Edge(v0, v1, EdgeState.TRAVERSABLE)]
    graphs = [Graph([v0, v1], edges=edges), Graph([v0, v1], edges=edges)]
    domain = [Edge(v0, v1, EdgeState.UNKNOWN), Edge(v1, v0, EdgeState.UNKNOWN)]
    baseline, optimized = await _run_find_safe_path_compare_modes(
        graphs=graphs,
        start=v0,
        goal=v1,
        T=2,
        V=2,
        port=5114,
        public_domain_edges=domain,
        mode="optimized",
    )
    assert baseline is None
    _assert_optimized_result(
        optimized,
        expect_sat=True,
        T=2,
        V=2,
        start=v0.id,
        goal=v1.id,
        domain_edges=domain,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )

import pytest

from private_path_query_logic import (
    ordered_symbols,
    directed_pairs_from_edges,
)
from private_path_query_utils import (
    Graph,
    Vertex,
    Edge,
    EdgeState,
)
from tests.test_private_path_query_logic import (
    _run_find_safe_path_compare_modes,
    _print_assignments_from_u_vector,
)


def _unknown_domain_edges_from_graph(graph: Graph) -> list[Edge]:
    return [
        Edge(e.vertex1, e.vertex2, EdgeState.UNKNOWN) for e in graph.to_directed_edges()
    ]


def _assert_optimized_result(
    optimized, expect_sat: bool, T: int, V: int, domain_edges: list[Edge]
):
    assert optimized is not None
    optimized_syms = ordered_symbols(
        T, V, directed_pairs=directed_pairs_from_edges(domain_edges, V)
    )
    if optimized.u_vector is not None:
        _print_assignments_from_u_vector(
            optimized.u_vector, T=T, V=V, symbols=optimized_syms
        )
        assert all(v in (0, 1) for v in optimized.u_vector)
    if expect_sat:
        assert optimized.is_solved
    else:
        assert not optimized.is_solved
    assert optimized.information_gain == 0.0


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
    _assert_optimized_result(optimized, expect_sat=True, T=1, V=2, domain_edges=domain)


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
    _assert_optimized_result(optimized, expect_sat=False, T=1, V=2, domain_edges=domain)


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
    _assert_optimized_result(optimized, expect_sat=True, T=2, V=3, domain_edges=domain)


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
    _assert_optimized_result(optimized, expect_sat=False, T=1, V=3, domain_edges=domain)


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
    _assert_optimized_result(optimized, expect_sat=True, T=2, V=4, domain_edges=domain)


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
    _assert_optimized_result(optimized, expect_sat=False, T=3, V=4, domain_edges=domain)


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
    _assert_optimized_result(optimized, expect_sat=True, T=2, V=2, domain_edges=domain)

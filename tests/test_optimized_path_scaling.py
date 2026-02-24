"""
Optimized-only path finding tests with scaling from 1 to 10 vertices.

Each test has:
- One clear traversable path from start to goal
- Occasional blocked edges (decoys that should not affect SAT)
- Only runs the optimized (edge-domain) solver
"""

import pytest
from private_path_query_logic import (
    directed_pairs_from_edges,
    ordered_symbols,
)
from private_path_query_utils import (
    Graph,
    Vertex,
    Edge,
    EdgeState,
)
from tests.private_path_query_test_helpers import (
    _unknown_domain_edges_from_graph,
    _run_find_safe_path_helper,
    _make_private_path_info,
    _assert_optimized_result,
)


def _make_linear_path_edges(n: int) -> list[Edge]:
    """
    Create a linear chain of traversable edges: 0 -> 1 -> 2 -> ... -> n-1.
    """
    vertices = [Vertex(id=i) for i in range(n)]
    edges = []
    for i in range(n - 1):
        edges.append(Edge(vertices[i], vertices[i + 1], EdgeState.TRAVERSABLE))
    return edges


def _make_graph_with_blocked_decoys(
    n: int,
    blocked_pairs: list[tuple[int, int]] | None = None,
) -> tuple[list[Vertex], list[Edge]]:
    """
    Create a graph with:
    - Linear traversable path: 0 -> 1 -> ... -> n-1
    - Optional blocked edges (decoys that don't affect the clear path)
    """
    vertices = [Vertex(id=i) for i in range(n)]
    edges = []

    # Main traversable path
    for i in range(n - 1):
        edges.append(Edge(vertices[i], vertices[i + 1], EdgeState.TRAVERSABLE))

    # Add blocked decoy edges
    if blocked_pairs:
        for u, v in blocked_pairs:
            if 0 <= u < n and 0 <= v < n and u != v:
                edges.append(Edge(vertices[u], vertices[v], EdgeState.BLOCKED))

    return vertices, edges


# =============================================================================
# V=2: Two vertices, one edge
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v2_direct_path():
    """V=2: Direct path 0 -> 1."""
    vertices, edges = _make_graph_with_blocked_decoys(n=2)
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 1, 2
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(1),
        port=6002,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=1,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=3: Three vertices, linear path
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v3_linear_path():
    """V=3: Path 0 -> 1 -> 2."""
    vertices, edges = _make_graph_with_blocked_decoys(n=3)
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 2, 3
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(2),
        port=6003,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=2,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=4: Four vertices with one blocked decoy
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v4_linear_with_blocked_decoy():
    """V=4: Path 0 -> 1 -> 2 -> 3, with blocked edge (0,3) as decoy."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=4,
        blocked_pairs=[(0, 3)],  # Shortcut is blocked, must use linear path
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 3, 4
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(3),
        port=6004,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=3,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=5: Five vertices with blocked decoys
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v5_linear_with_blocked_decoys():
    """V=5: Path 0 -> 1 -> 2 -> 3 -> 4, with blocked edges as decoys."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=5,
        blocked_pairs=[(0, 4), (1, 3)],  # Shortcuts are blocked
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 4, 5
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(4),
        port=6005,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=4,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=6: Six vertices
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v6_linear_path():
    """V=6: Path 0 -> 1 -> 2 -> 3 -> 4 -> 5."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=6,
        blocked_pairs=[(0, 5), (2, 4)],
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 5, 6
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(5),
        port=6006,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=5,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=7: Seven vertices
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v7_linear_path():
    """V=7: Path 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=7,
        blocked_pairs=[(0, 6), (1, 5), (2, 4)],
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 6, 7
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(6),
        port=6007,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=6,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=8: Eight vertices (power of 2 - no range restriction clauses)
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v8_linear_path():
    """V=8: Path 0 -> ... -> 7. Power of 2 means no range restriction clauses."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=8,
        blocked_pairs=[(0, 7), (1, 6), (3, 5)],
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 7, 8
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(7),
        port=6008,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=7,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=9: Nine vertices (not power of 2 - has range restriction clauses)
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v9_linear_path():
    """V=9: Path 0 -> ... -> 8. Not power of 2, so has range clauses for codes 9-15."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=9,
        blocked_pairs=[(0, 8), (2, 6), (3, 7)],
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 8, 9
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(8),
        port=6009,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=8,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# V=10: Ten vertices
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v10_linear_path():
    """V=10: Path 0 -> ... -> 9."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=10,
        blocked_pairs=[(0, 9), (1, 8), (2, 7), (4, 6)],
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 9, 10
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(9),
        port=6010,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=9,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


# =============================================================================
# Additional edge cases
# =============================================================================


@pytest.mark.asyncio
async def test_optimized_v3_with_extra_horizon():
    """V=3: Path 0 -> 1 -> 2 with T=4 (allows waiting)."""
    vertices, edges = _make_graph_with_blocked_decoys(n=3)
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 4, 3  # T=4 gives extra steps allowing waiting
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(2),
        port=6011,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=0,
        goal=2,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


@pytest.mark.asyncio
async def test_optimized_v4_start_equals_goal():
    """V=4: start == goal with T=0 (trivially SAT)."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=4,
        blocked_pairs=[(0, 2), (1, 3)],
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 0, 4
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(2),
        goal=Vertex(2),
        port=6012,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=2,
        goal=2,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )


@pytest.mark.asyncio
async def test_optimized_v5_partial_path():
    """V=5: Path from 1 -> 2 -> 3 (not starting at 0)."""
    vertices, edges = _make_graph_with_blocked_decoys(
        n=5,
        blocked_pairs=[(0, 4)],
    )
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]
    domain_edges = _unknown_domain_edges_from_graph(graphs[0])
    T, V = 2, 5
    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(1),
        goal=Vertex(3),
        port=6013,
    )

    _assert_optimized_result(
        result,
        expect_sat=True,
        T=info.T,
        V=info.V,
        start=1,
        goal=3,
        domain_edges=domain_edges,
        bob_edges_by_party=[g.to_directed_edges() for g in graphs],
    )

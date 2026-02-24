"""
Tests for weighted edge functionality in private path query.

These tests verify that when the solver must relax a soft constraint (blocked edge),
it prefers to relax edges with lower weights.

Scenario:
- 3 vertices: 0, 1, 2
- Start: 0, Goal: 2
- Edge 0->1: TRAVERSABLE
- Edge 1->2: BLOCKED (option A)
- Edge 0->2: BLOCKED (option B)

No fully traversable path exists, so the solver must relax one blocked edge.
The edge with lower weight should be the one relaxed (used in the final path).
"""

import pytest
from private_path_query_logic import (
    directed_pairs_from_edges,
    ordered_symbols,
    decode_path_from_assignment,
    assignment_from_u_vector,
)
from private_path_query_utils import (
    Graph,
    Vertex,
    Edge,
    EdgeState,
    normalize_edge_weights,
)
from tests.private_path_query_test_helpers import (
    _unknown_domain_edges_from_graph,
    _run_find_safe_path_helper,
    _make_private_path_info,
    _assert_optimized_result,
)


def _decode_path_from_result(result, T: int, V: int, symbols=None) -> list[int]:
    """Extract the vertex path from a computation result."""
    if result is None or result.u_vector is None:
        return []
    model = assignment_from_u_vector(
        u_vector=result.u_vector, T=T, V=V, symbols=symbols
    )
    return decode_path_from_assignment(model, T=T, V=V)


# =============================================================================
# Test: Lower weight on edge 1->2 should make solver use path 0->1->2
# =============================================================================


@pytest.mark.asyncio
async def test_weighted_prefer_lower_weight_edge_A():
    """
    When edge 1->2 has lower weight than edge 0->2,
    the solver should prefer path 0->1->2 (relaxing the lower-weight blocked edge).

    Setup:
    - Edge 0->1: TRAVERSABLE (weight doesn't matter for traversable)
    - Edge 1->2: BLOCKED, weight=0.1 (LOW - should be relaxed)
    - Edge 0->2: BLOCKED, weight=0.9 (HIGH - should NOT be relaxed)
    """
    vertices = [Vertex(i) for i in range(3)]

    # Create edges with weights - lower weight on 1->2
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE, weight=1.0),
        Edge(
            vertices[1], vertices[2], EdgeState.BLOCKED, weight=0.1
        ),  # LOW - prefer this
        Edge(
            vertices[0], vertices[2], EdgeState.BLOCKED, weight=0.9
        ),  # HIGH - avoid this
    ]

    # Edge weights are automatically normalized in build_bob_q
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]

    domain_edges = _unknown_domain_edges_from_graph(graphs[0])

    V, T = 3, 2  # T=2 allows path 0->1->2

    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
        weighted=True,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(2),
        port=6101,
        weighted=True,
    )

    # The result should be SAT (a path was found)
    assert result is not None, "Expected a result"
    assert result.u_vector is not None, "Expected u_vector in result"

    # Decode the path
    path = _decode_path_from_result(result, T=info.T, V=info.V)
    print(f"Decoded path: {path}")

    # The path should be 0 -> 1 -> 2 (using the lower-weight blocked edge 1->2)
    # This means vertex 1 should appear in the path
    assert 1 in path, f"Expected path through vertex 1 (0->1->2), but got {path}"
    assert path[0] == 0, f"Path should start at 0, got {path}"
    assert path[-1] == 2, f"Path should end at 2, got {path}"


# =============================================================================
# Test: Lower weight on edge 0->2 should make solver use path 0->2
# =============================================================================


@pytest.mark.asyncio
async def test_weighted_prefer_lower_weight_edge_B():
    """
    When edge 0->2 has lower weight than edge 1->2,
    the solver should prefer path 0->2 (relaxing the lower-weight blocked edge).

    Setup:
    - Edge 0->1: TRAVERSABLE (weight doesn't matter for traversable)
    - Edge 1->2: BLOCKED, weight=0.9 (HIGH - should NOT be relaxed)
    - Edge 0->2: BLOCKED, weight=0.1 (LOW - should be relaxed)
    """
    vertices = [Vertex(i) for i in range(3)]

    # Create edges with weights - lower weight on 0->2
    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE, weight=1.0),
        Edge(
            vertices[1], vertices[2], EdgeState.BLOCKED, weight=0.9
        ),  # HIGH - avoid this
        Edge(
            vertices[0], vertices[2], EdgeState.BLOCKED, weight=0.1
        ),  # LOW - prefer this
    ]

    # Edge weights are automatically normalized in build_bob_q
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]

    domain_edges = _unknown_domain_edges_from_graph(graphs[0])

    V, T = 3, 1  # T=1 is enough for direct path 0->2

    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
        weighted=True,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(2),
        port=6102,
        weighted=True,
    )

    # The result should be SAT (a path was found)
    assert result is not None, "Expected a result"
    assert result.u_vector is not None, "Expected u_vector in result"

    # Decode the path
    path = _decode_path_from_result(result, T=info.T, V=info.V)
    print(f"Decoded path: {path}")

    # The path should be 0 -> 2 (direct, using the lower-weight blocked edge 0->2)
    # This means vertex 1 should NOT appear in the path (or the path is just [0, 2])
    assert path[0] == 0, f"Path should start at 0, got {path}"
    assert path[-1] == 2, f"Path should end at 2, got {path}"
    # With T=1, the path should be exactly [0, 2]
    assert path == [0, 2], f"Expected direct path [0, 2], but got {path}"


# =============================================================================
# Test: Extreme weight difference should clearly prefer low-weight edge
# =============================================================================


@pytest.mark.asyncio
async def test_weighted_extreme_difference():
    """
    With extreme weight difference (0.01 vs 0.99), the solver should
    strongly prefer the low-weight edge.

    Setup:
    - Edge 0->1: TRAVERSABLE
    - Edge 1->2: BLOCKED, weight=0.01 (VERY LOW)
    - Edge 0->2: BLOCKED, weight=0.99 (VERY HIGH)
    """
    vertices = [Vertex(i) for i in range(3)]

    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE, weight=1.0),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED, weight=0.01),  # VERY LOW
        Edge(vertices[0], vertices[2], EdgeState.BLOCKED, weight=0.99),  # VERY HIGH
    ]

    # Print edge weights for debugging (will be normalized in build_bob_q)
    print("Edge weights (before auto-normalization):")
    for e in edges:
        print(f"  {e.vertex1.id}->{e.vertex2.id}: {e.state.name}, weight={e.weight}")

    # Edge weights are automatically normalized in build_bob_q
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]

    domain_edges = _unknown_domain_edges_from_graph(graphs[0])

    V, T = 3, 2

    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
        weighted=True,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(2),
        port=6103,
        weighted=True,
    )

    assert result is not None, "Expected a result"
    assert result.u_vector is not None, "Expected u_vector in result"

    path = _decode_path_from_result(result, T=info.T, V=info.V)
    print(f"Decoded path: {path}")

    # Should use path 0->1->2 (through vertex 1)
    assert 1 in path, f"Expected path through vertex 1, but got {path}"


# =============================================================================
# Test: Equal weights - either path is acceptable
# =============================================================================


@pytest.mark.asyncio
async def test_weighted_equal_weights():
    """
    When both blocked edges have equal weight, either path is acceptable.
    Just verify that a valid path is found.
    """
    vertices = [Vertex(i) for i in range(3)]

    edges = [
        Edge(vertices[0], vertices[1], EdgeState.TRAVERSABLE, weight=1.0),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED, weight=0.5),  # EQUAL
        Edge(vertices[0], vertices[2], EdgeState.BLOCKED, weight=0.5),  # EQUAL
    ]

    # Edge weights are automatically normalized in build_bob_q
    graphs = [
        Graph(vertices=vertices, edges=edges),
        Graph(vertices=vertices, edges=edges),
    ]

    domain_edges = _unknown_domain_edges_from_graph(graphs[0])

    V, T = 3, 2

    info = _make_private_path_info(
        T=T,
        V=V,
        num_bobs=len(graphs),
        use_edge_domain=True,
        edge_domain_edges=domain_edges,
        weighted=True,
    )

    result = await _run_find_safe_path_helper(
        info=info,
        graphs=graphs,
        start=Vertex(0),
        goal=Vertex(2),
        port=6104,
        weighted=True,
    )

    assert result is not None, "Expected a result"
    assert result.u_vector is not None, "Expected u_vector in result"

    path = _decode_path_from_result(result, T=info.T, V=info.V)
    print(f"Decoded path: {path}")

    # Either [0, 2] or [0, 1, 2] is valid
    assert path[0] == 0, f"Path should start at 0, got {path}"
    assert path[-1] == 2, f"Path should end at 2, got {path}"


# =============================================================================
# Test: Verify edge weights are properly normalized
# =============================================================================


def test_edge_weights_normalized_to_budget():
    """
    Verify that normalize_edge_weights correctly normalizes weights to sum to 1.0.

    Test cases:
    1. All explicit weights -> scaled to sum to budget
    2. Mix of explicit and implicit (None) weights -> all normalized
    3. All implicit weights -> equal distribution
    4. Custom budget value
    """
    vertices = [Vertex(i) for i in range(5)]

    # Test 1: All explicit weights
    edges_explicit = [
        Edge(vertices[0], vertices[1], EdgeState.BLOCKED, weight=2.0),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED, weight=3.0),
        Edge(vertices[2], vertices[3], EdgeState.BLOCKED, weight=5.0),
    ]
    normalized_explicit = normalize_edge_weights(edges_explicit, total_budget=1.0)
    total_explicit = sum(e.weight for e in normalized_explicit)
    assert (
        abs(total_explicit - 1.0) < 1e-9
    ), f"Explicit weights should sum to 1.0, got {total_explicit}"
    # Check relative proportions are preserved: 2:3:5 ratio
    assert abs(normalized_explicit[0].weight - 0.2) < 1e-9, "Weight ratio not preserved"
    assert abs(normalized_explicit[1].weight - 0.3) < 1e-9, "Weight ratio not preserved"
    assert abs(normalized_explicit[2].weight - 0.5) < 1e-9, "Weight ratio not preserved"
    print(f"✓ Test 1 (explicit): weights sum to {total_explicit:.4f}")

    # Test 2: Mix of explicit and implicit weights
    edges_mixed = [
        Edge(vertices[0], vertices[1], EdgeState.BLOCKED, weight=1.0),  # Explicit
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),  # Implicit (default=1.0)
        Edge(vertices[2], vertices[3], EdgeState.BLOCKED, weight=2.0),  # Explicit
    ]
    normalized_mixed = normalize_edge_weights(edges_mixed, total_budget=1.0)
    total_mixed = sum(e.weight for e in normalized_mixed)
    assert (
        abs(total_mixed - 1.0) < 1e-9
    ), f"Mixed weights should sum to 1.0, got {total_mixed}"
    # With default_weight=1.0: raw weights are [1, 1, 2] -> normalized [0.25, 0.25, 0.5]
    assert (
        abs(normalized_mixed[0].weight - 0.25) < 1e-9
    ), "Mixed weight ratio not preserved"
    assert (
        abs(normalized_mixed[1].weight - 0.25) < 1e-9
    ), "Mixed weight ratio not preserved"
    assert (
        abs(normalized_mixed[2].weight - 0.5) < 1e-9
    ), "Mixed weight ratio not preserved"
    print(f"✓ Test 2 (mixed): weights sum to {total_mixed:.4f}")

    # Test 3: All implicit weights (None) -> equal distribution
    edges_implicit = [
        Edge(vertices[0], vertices[1], EdgeState.BLOCKED),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED),
        Edge(vertices[2], vertices[3], EdgeState.BLOCKED),
        Edge(vertices[3], vertices[4], EdgeState.BLOCKED),
    ]
    normalized_implicit = normalize_edge_weights(edges_implicit, total_budget=1.0)
    total_implicit = sum(e.weight for e in normalized_implicit)
    assert (
        abs(total_implicit - 1.0) < 1e-9
    ), f"Implicit weights should sum to 1.0, got {total_implicit}"
    # All should be equal: 1/4 = 0.25
    for e in normalized_implicit:
        assert (
            abs(e.weight - 0.25) < 1e-9
        ), f"Expected equal weight 0.25, got {e.weight}"
    print(f"✓ Test 3 (implicit): weights sum to {total_implicit:.4f}, all equal")

    # Test 4: Custom budget value
    edges_custom_budget = [
        Edge(vertices[0], vertices[1], EdgeState.BLOCKED, weight=1.0),
        Edge(vertices[1], vertices[2], EdgeState.BLOCKED, weight=1.0),
    ]
    custom_budget = 2.5
    normalized_custom = normalize_edge_weights(
        edges_custom_budget, total_budget=custom_budget
    )
    total_custom = sum(e.weight for e in normalized_custom)
    assert (
        abs(total_custom - custom_budget) < 1e-9
    ), f"Custom budget should sum to {custom_budget}, got {total_custom}"
    print(f"✓ Test 4 (custom budget): weights sum to {total_custom:.4f}")

    print("\n✓ All normalization tests passed!")

"""
Tests for PRM (Probabilistic Roadmap) benchmarking with private path queries.
Supports varying percentages of blocked edges with randomized safe paths.

Run with: pytest tests/test_prm_benchmarking.py -v
Tests run serially (one at a time) via asyncio lock.
"""

import asyncio
import pytest
from private_path_query_utils import (
    Graph,
    Vertex,
)
from tests.private_path_query_test_helpers import (
    _unknown_domain_edges_from_graph,
    _run_find_safe_path_helper,
    _make_private_path_info,
)
from tests.prm_graph_configs import (
    GRAPH_CONFIGS,
    BLOCKING_PERCENTAGES,
    create_edges_with_random_safe_path,
)

# Global lock to ensure tests run one at a time
_test_lock = asyncio.Lock()


async def _run_prm_test(
    graph_name: str,
    block_percent: int,
    port: int,
    seed: int = 42,
):
    """Run a PRM test with specified blocking percentage and randomized safe path."""
    async with _test_lock:
        config = GRAPH_CONFIGS[graph_name]
        num_nodes = config["num_nodes"]
        edge_pairs = config["edge_pairs"]
        start_id = config["start_id"]
        goal_id = config["goal_id"]
        max_path_length = config["max_path_length"]

        vertices = [Vertex(id=i) for i in range(num_nodes)]

        # Create edges with randomized safe path
        edges, safe_path_nodes, safe_path_edges = create_edges_with_random_safe_path(
            vertices=vertices,
            edge_pairs=edge_pairs,
            start_id=start_id,
            goal_id=goal_id,
            max_path_length=max_path_length,
            block_percent=block_percent,
            seed=seed,
        )

        safe_path_str = " -> ".join(str(n) for n in safe_path_nodes)
        print(f"\n{'='*60}")
        print(f"Graph: {graph_name}, Blocking: {block_percent}%")
        print(f"Safe path: {safe_path_str} ({len(safe_path_edges)} edges)")
        print(f"{'='*60}")

        graphs = [
            Graph(vertices=vertices, edges=edges),
            Graph(vertices=vertices, edges=edges),
        ]

        domain_edges = _unknown_domain_edges_from_graph(graphs[0])
        
        # T should be at least as long as the safe path
        T = len(safe_path_edges)
        
        info = _make_private_path_info(
            T=T,
            V=num_nodes,
            num_bobs=len(graphs),
            use_edge_domain=True,
            edge_domain_edges=domain_edges,
        )

        # Create test name with blocking percentage
        test_name = f"prm_{graph_name}_blocked_{block_percent}pct"

        result = await _run_find_safe_path_helper(
            info=info,
            graphs=graphs,
            start=Vertex(start_id),
            goal=Vertex(goal_id),
            port=port,
            test_name=test_name,
        )

        assert result is not None
        assert result.is_solved, f"SAT solver should find path for {graph_name} at {block_percent}% blocking"


# =============================================================================
# Graph 1: 5 nodes, 6 edges
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_5_nodes(block_percent):
    """PRM graph: 5 nodes, 6 edges with randomized safe path."""
    await _run_prm_test(
        graph_name="5v_6e",
        block_percent=block_percent,
        port=7010 + block_percent,
    )


# =============================================================================
# Graph 2: 6 nodes, 7 edges
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_6_nodes(block_percent):
    """PRM graph: 6 nodes, 7 edges with randomized safe path."""
    await _run_prm_test(
        graph_name="6v_7e",
        block_percent=block_percent,
        port=7020 + block_percent,
    )


# =============================================================================
# Graph 3: 7 nodes, 13 edges
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_7_nodes(block_percent):
    """PRM graph: 7 nodes, 13 edges with randomized safe path."""
    await _run_prm_test(
        graph_name="7v_13e",
        block_percent=block_percent,
        port=7030 + block_percent,
    )


# =============================================================================
# Graph 4: 8 nodes, 21 edges
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_8_nodes(block_percent):
    """PRM graph: 8 nodes, 21 edges with randomized safe path."""
    await _run_prm_test(
        graph_name="8v_21e",
        block_percent=block_percent,
        port=7040 + block_percent,
    )


# =============================================================================
# Graph 5: 9 nodes, 19 edges
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_9_nodes(block_percent):
    """PRM graph: 9 nodes, 19 edges with randomized safe path."""
    await _run_prm_test(
        graph_name="9v_19e",
        block_percent=block_percent,
        port=7050 + block_percent,
    )


# =============================================================================
# Graph 6: 10 nodes, 25 edges
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_10_nodes(block_percent):
    """PRM graph: 10 nodes, 25 edges with randomized safe path."""
    await _run_prm_test(
        graph_name="10v_25e",
        block_percent=block_percent,
        port=7060 + block_percent,
    )

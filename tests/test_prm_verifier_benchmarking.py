"""
Tests for PRM (Probabilistic Roadmap) benchmarking with iterative path queries.
Compares three path-finding strategies: entropy, likelihood, and random.
Logs results to CSV including iterations and information gain per iteration.
"""

import asyncio
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

import pytest
from private_path_query_utils import (
    Graph,
    GraphPath,
    Vertex,
    Edge,
    EdgeState,
)
from bayesien_paths import BayesGraphMap
from tests.test_private_path_query_utils import (
    _run_verifier_test_helper,
)
from tests.prm_graph_configs import (
    GRAPH_CONFIGS,
    BLOCKING_PERCENTAGES,
    create_edges_with_random_safe_path,
)


# =============================================================================
# Configuration
# =============================================================================

CSV_OUTPUT_DIR = Path(__file__).parent.parent / "trace-artifacts" / "csv" / "verifier"
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITERATIONS = 20
P_INIT = 0.3  # Initial probability that each edge is blocked

# Ensure tests run one at a time
_test_lock = asyncio.Lock()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class IterationLog:
    """Log of a single iteration."""

    iteration: int
    path_str: str
    is_safe: bool
    information_gain: float
    cumulative_info_gain: float


@dataclass
class StrategyResult:
    """Result of running a single strategy."""

    strategy: str
    found_safe_path: bool
    iterations_used: int
    total_info_gain: float
    iteration_logs: List[IterationLog] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of a full benchmark run."""

    graph_name: str
    num_nodes: int
    num_edges: int
    block_percent: int
    safe_path_str: str = ""
    safe_path_length: int = 0
    results: Dict[str, StrategyResult] = field(default_factory=dict)


# =============================================================================
# Helper Functions
# =============================================================================


def _create_alice_graph(
    vertices: List[Vertex], edge_pairs: List[Tuple[int, int]]
) -> Graph:
    """Create Alice's belief graph with all edges as TRAVERSABLE."""
    edges = [
        Edge(vertices[v1], vertices[v2], EdgeState.TRAVERSABLE) for v1, v2 in edge_pairs
    ]
    return Graph(vertices=vertices, edges=edges)


def _path_to_str(start_id: int, path: Optional[GraphPath]) -> str:
    """Convert a GraphPath to a string representation."""
    if path is None:
        return "NO_PATH"
    return " -> ".join([str(start_id)] + [str(e.vertex2.id) for e in path.moves])


# =============================================================================
# Strategy Runner
# =============================================================================


async def _run_strategy(
    strategy_name: str,
    path_finder: Callable,
    graphs: List[Graph],
    alice_graph: Graph,
    start: Vertex,
    goal: Vertex,
    max_path_length: int,
    max_iterations: int = MAX_ITERATIONS,
) -> StrategyResult:
    """
    Run a single path-finding strategy for multiple iterations.

    Args:
        strategy_name: Name of the strategy (for logging)
        path_finder: Function that takes (belief_map, start, goal, max_length) and returns (score, path)
        graphs: Ground truth graphs for Bobs
        alice_graph: Alice's belief graph (topology only)
        start: Starting vertex
        goal: Goal vertex
        max_path_length: Maximum edges in path
        max_iterations: Maximum iterations to try

    Returns:
        StrategyResult with logs of all iterations
    """
    alice_belief = BayesGraphMap(alice_graph, p_init=P_INIT)

    iteration_logs = []
    total_info_gain = 0.0
    found_safe = False
    iterations_used = 0

    for iteration in range(max_iterations):
        # Get path using the strategy
        _, path = path_finder(alice_belief, start, goal, max_path_length)

        if path is None:
            break

        path_str = _path_to_str(start.id, path)

        # Run verifier
        result = await _run_verifier_test_helper(graphs, path, is_graph=True)
        is_safe = result.is_solved
        info_gain = result.information_gain
        total_info_gain += info_gain

        iteration_logs.append(
            IterationLog(
                iteration=iteration + 1,
                path_str=path_str,
                is_safe=is_safe,
                information_gain=info_gain,
                cumulative_info_gain=total_info_gain,
            )
        )

        # Update belief state
        alice_belief.update_probabilities(path, safe=is_safe)
        iterations_used = iteration + 1

        # Check if we found a viable path
        if alice_belief.check_viable_path(start, goal):
            found_safe = True
            break

    return StrategyResult(
        strategy=strategy_name,
        found_safe_path=found_safe,
        iterations_used=iterations_used,
        total_info_gain=total_info_gain,
        iteration_logs=iteration_logs,
    )


async def _run_all_strategies(
    graph_name: str,
    config: Dict[str, Any],
    block_percent: int,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Run all three strategies on a single graph configuration.

    Returns:
        BenchmarkResult with results for all strategies
    """
    num_nodes = config["num_nodes"]
    edge_pairs = config["edge_pairs"]
    start_id = config["start_id"]
    goal_id = config["goal_id"]
    max_path_length = config["max_path_length"]

    vertices = [Vertex(id=i) for i in range(num_nodes)]

    # Create ground truth edges with a randomly selected safe path
    ground_truth_edges, safe_path_nodes, safe_path_edges = (
        create_edges_with_random_safe_path(
            vertices=vertices,
            edge_pairs=edge_pairs,
            start_id=start_id,
            goal_id=goal_id,
            max_path_length=max_path_length,
            block_percent=block_percent,
            seed=seed,
        )
    )

    # Create graphs for Bobs (2 Bobs with identical graphs)
    graphs = [
        Graph(vertices=vertices, edges=ground_truth_edges),
        Graph(vertices=vertices, edges=ground_truth_edges),
    ]

    # Create Alice's belief graph (knows topology, not states)
    alice_graph = _create_alice_graph(vertices, edge_pairs)

    start = vertices[start_id]
    goal = vertices[goal_id]

    safe_path_str = " -> ".join(str(n) for n in safe_path_nodes)

    result = BenchmarkResult(
        graph_name=graph_name,
        num_nodes=num_nodes,
        num_edges=len(edge_pairs),
        block_percent=block_percent,
        safe_path_str=safe_path_str,
        safe_path_length=len(safe_path_edges),
    )

    # Define strategy functions
    strategies = {
        "entropy": lambda bm, s, g, ml: bm.find_highest_entropy_path(s, g, ml),
        "likelihood": lambda bm, s, g, ml: bm.find_highes_likelihood_safe_path(
            s, g, ml
        ),
        "random": lambda bm, s, g, ml: bm.find_random_path(s, g, ml),
    }
    print(f"\n{'='*60}")
    print(f"Graph: {graph_name}, Blocking: {block_percent}%")
    print(f"Safe path: {safe_path_str} ({len(safe_path_edges)} edges)")
    print(f"{'='*60}")

    for strategy_name, path_finder in strategies.items():
        print(f"\n--- Strategy: {strategy_name} ---")

        strategy_result = await _run_strategy(
            strategy_name=strategy_name,
            path_finder=path_finder,
            graphs=graphs,
            alice_graph=alice_graph,
            start=start,
            goal=goal,
            max_path_length=max_path_length,
        )

        result.results[strategy_name] = strategy_result

        status = "✓ FOUND" if strategy_result.found_safe_path else "✗ NOT FOUND"
        print(f"  {status} in {strategy_result.iterations_used} iterations")
        print(f"  Total info gain: {strategy_result.total_info_gain:.4f}")

    return result


# =============================================================================
# CSV Logging
# =============================================================================


def _write_single_result_csv(result: BenchmarkResult):
    """Write a single benchmark result to CSV files (overwrites existing)."""
    summary_file = (
        CSV_OUTPUT_DIR
        / f"verifier_{result.graph_name}_blocked_{result.block_percent}pct.csv"
    )
    iterations_file = (
        CSV_OUTPUT_DIR
        / f"verifier_{result.graph_name}_blocked_{result.block_percent}pct_iterations.csv"
    )

    # Write summary CSV (overwrite)
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "graph_name",
                "num_nodes",
                "num_edges",
                "block_percent",
                "safe_path",
                "safe_path_length",
                "strategy",
                "found_safe_path",
                "iterations_used",
                "total_info_gain",
            ]
        )

        for strategy_name, strategy_result in result.results.items():
            writer.writerow(
                [
                    result.graph_name,
                    result.num_nodes,
                    result.num_edges,
                    result.block_percent,
                    result.safe_path_str,
                    result.safe_path_length,
                    strategy_name,
                    strategy_result.found_safe_path,
                    strategy_result.iterations_used,
                    f"{strategy_result.total_info_gain:.6f}",
                ]
            )

    print(f"\nSummary CSV written to: {summary_file}")

    # Write detailed iteration CSV (overwrite)
    with open(iterations_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "graph_name",
                "num_nodes",
                "num_edges",
                "block_percent",
                "safe_path",
                "safe_path_length",
                "strategy",
                "iteration",
                "path",
                "is_safe",
                "info_gain",
                "cumulative_info_gain",
            ]
        )

        for strategy_name, strategy_result in result.results.items():
            for log in strategy_result.iteration_logs:
                writer.writerow(
                    [
                        result.graph_name,
                        result.num_nodes,
                        result.num_edges,
                        result.block_percent,
                        result.safe_path_str,
                        result.safe_path_length,
                        strategy_name,
                        log.iteration,
                        log.path_str,
                        log.is_safe,
                        f"{log.information_gain:.6f}",
                        f"{log.cumulative_info_gain:.6f}",
                    ]
                )

    print(f"Iteration CSV written to: {iterations_file}")


def _write_benchmark_csv(results: List[BenchmarkResult], filename: str):
    """Write benchmark results to CSV file."""
    filepath = CSV_OUTPUT_DIR / filename

    # Write summary CSV
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "graph_name",
                "num_nodes",
                "num_edges",
                "block_percent",
                "strategy",
                "found_safe_path",
                "iterations_used",
                "total_info_gain",
            ]
        )

        for result in results:
            for strategy_name, strategy_result in result.results.items():
                writer.writerow(
                    [
                        result.graph_name,
                        result.num_nodes,
                        result.num_edges,
                        result.block_percent,
                        strategy_name,
                        strategy_result.found_safe_path,
                        strategy_result.iterations_used,
                        f"{strategy_result.total_info_gain:.6f}",
                    ]
                )

    print(f"\nSummary CSV written to: {filepath}")

    # Write detailed iteration CSV
    detail_filepath = CSV_OUTPUT_DIR / filename.replace(".csv", "_iterations.csv")
    with open(detail_filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "graph_name",
                "num_nodes",
                "num_edges",
                "block_percent",
                "strategy",
                "iteration",
                "path",
                "is_safe",
                "info_gain",
                "cumulative_info_gain",
            ]
        )

        for result in results:
            for strategy_name, strategy_result in result.results.items():
                for log in strategy_result.iteration_logs:
                    writer.writerow(
                        [
                            result.graph_name,
                            result.num_nodes,
                            result.num_edges,
                            result.block_percent,
                            strategy_name,
                            log.iteration,
                            log.path_str,
                            log.is_safe,
                            f"{log.information_gain:.6f}",
                            f"{log.cumulative_info_gain:.6f}",
                        ]
                    )

    print(f"Iteration CSV written to: {detail_filepath}")


# =============================================================================
# Test Functions
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_5_nodes_all_strategies(block_percent):
    """Benchmark all strategies on 5-node graph."""
    async with _test_lock:
        result = await _run_all_strategies(
            "5v_6e", GRAPH_CONFIGS["5v_6e"], block_percent
        )

        # Write results to CSV
        _write_single_result_csv(result)

        # All strategies must find a path (path edges are never blocked)
        for strategy_name, strategy_result in result.results.items():
            assert strategy_result.found_safe_path, (
                f"Strategy '{strategy_name}' failed to find safe path "
                f"for 5v_6e at {block_percent}% blocking"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_6_nodes_all_strategies(block_percent):
    """Benchmark all strategies on 6-node graph."""
    async with _test_lock:
        result = await _run_all_strategies(
            "6v_7e", GRAPH_CONFIGS["6v_7e"], block_percent
        )

        # Write results to CSV
        _write_single_result_csv(result)

        # All strategies must find a path
        for strategy_name, strategy_result in result.results.items():
            assert strategy_result.found_safe_path, (
                f"Strategy '{strategy_name}' failed to find safe path "
                f"for 6v_7e at {block_percent}% blocking"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_7_nodes_all_strategies(block_percent):
    """Benchmark all strategies on 7-node graph."""
    async with _test_lock:
        result = await _run_all_strategies(
            "7v_13e", GRAPH_CONFIGS["7v_13e"], block_percent
        )

        # Write results to CSV
        _write_single_result_csv(result)

        # All strategies must find a path
        for strategy_name, strategy_result in result.results.items():
            assert strategy_result.found_safe_path, (
                f"Strategy '{strategy_name}' failed to find safe path "
                f"for 7v_13e at {block_percent}% blocking"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_8_nodes_all_strategies(block_percent):
    """Benchmark all strategies on 8-node graph."""
    async with _test_lock:
        result = await _run_all_strategies(
            "8v_21e", GRAPH_CONFIGS["8v_21e"], block_percent
        )

        # Write results to CSV
        _write_single_result_csv(result)

        # All strategies must find a path
        for strategy_name, strategy_result in result.results.items():
            assert strategy_result.found_safe_path, (
                f"Strategy '{strategy_name}' failed to find safe path "
                f"for 8v_21e at {block_percent}% blocking"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_9_nodes_all_strategies(block_percent):
    """Benchmark all strategies on 9-node graph."""
    async with _test_lock:
        result = await _run_all_strategies(
            "9v_19e", GRAPH_CONFIGS["9v_19e"], block_percent
        )

        # Write results to CSV
        _write_single_result_csv(result)

        # All strategies must find a path
        for strategy_name, strategy_result in result.results.items():
            assert strategy_result.found_safe_path, (
                f"Strategy '{strategy_name}' failed to find safe path "
                f"for 9v_19e at {block_percent}% blocking"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("block_percent", BLOCKING_PERCENTAGES)
async def test_prm_10_nodes_all_strategies(block_percent):
    """Benchmark all strategies on 10-node graph."""
    async with _test_lock:
        result = await _run_all_strategies(
            "10v_25e", GRAPH_CONFIGS["10v_25e"], block_percent
        )

        # Write results to CSV
        _write_single_result_csv(result)

        # All strategies must find a path
        for strategy_name, strategy_result in result.results.items():
            assert strategy_result.found_safe_path, (
                f"Strategy '{strategy_name}' failed to find safe path "
                f"for 10v_25e at {block_percent}% blocking"
            )


# =============================================================================
# Full Benchmark Suite (runs all graphs and blocking percentages)
# =============================================================================


@pytest.mark.asyncio
async def test_full_benchmark_suite():
    """
    Run full benchmark suite across all graphs and blocking percentages.
    Writes results to CSV for analysis.
    """
    async with _test_lock:
        all_results = []

        for graph_name, config in GRAPH_CONFIGS.items():
            for block_percent in BLOCKING_PERCENTAGES:
                result = await _run_all_strategies(graph_name, config, block_percent)
                all_results.append(result)

        # Write all results to CSV
        _write_benchmark_csv(all_results, "verifier_strategy_comparison.csv")

        # Print summary table
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(
            f"{'Graph':<10} {'Block%':<8} {'Entropy':<15} {'Likelihood':<15} {'Random':<15}"
        )
        print("-" * 80)

        for result in all_results:
            entropy_str = f"{'✓' if result.results['entropy'].found_safe_path else '✗'} ({result.results['entropy'].iterations_used})"
            likelihood_str = f"{'✓' if result.results['likelihood'].found_safe_path else '✗'} ({result.results['likelihood'].iterations_used})"
            random_str = f"{'✓' if result.results['random'].found_safe_path else '✗'} ({result.results['random'].iterations_used})"

            print(
                f"{result.graph_name:<10} {result.block_percent:<8} {entropy_str:<15} {likelihood_str:<15} {random_str:<15}"
            )

        print("=" * 80)

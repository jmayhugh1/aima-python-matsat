"""
Shared PRM graph configurations and helper functions for benchmarking tests.
Used by both SAT solver (test_prm_benchmarking.py) and verifier (test_prm_verifier_benchmarking.py).
"""

import random
from typing import List, Tuple, Set, Dict
from private_path_query_utils import Vertex, Edge, EdgeState


# =============================================================================
# Graph Definitions
# =============================================================================

# Each graph is defined by its topology only - safe path is found randomly at runtime
GRAPH_CONFIGS = {
    "5v_6e": {
        "num_nodes": 5,
        "edge_pairs": [(0, 3), (0, 1), (0, 2), (2, 4), (2, 3), (3, 4)],
        "start_id": 0,
        "goal_id": 4,
        "max_path_length": 4,
    },
    "6v_7e": {
        "num_nodes": 6,
        "edge_pairs": [(0, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5)],
        "start_id": 0,
        "goal_id": 5,
        "max_path_length": 5,
    },
    "7v_13e": {
        "num_nodes": 7,
        "edge_pairs": [
            (0, 3),
            (0, 2),
            (1, 5),
            (1, 6),
            (1, 4),
            (1, 2),
            (1, 3),
            (2, 4),
            (2, 5),
            (3, 5),
            (3, 4),
            (4, 5),
            (5, 6),
        ],
        "start_id": 0,
        "goal_id": 6,
        "max_path_length": 5,
    },
    "8v_21e": {
        "num_nodes": 8,
        "edge_pairs": [
            (0, 3),
            (0, 1),
            (0, 2),
            (0, 6),
            (0, 5),
            (0, 4),
            (1, 6),
            (1, 2),
            (1, 3),
            (1, 5),
            (1, 7),
            (2, 6),
            (2, 3),
            (2, 5),
            (2, 7),
            (3, 6),
            (3, 5),
            (3, 4),
            (3, 7),
            (4, 7),
            (4, 5),
        ],
        "start_id": 0,
        "goal_id": 7,
        "max_path_length": 5,
    },
    "9v_19e": {
        "num_nodes": 9,
        "edge_pairs": [
            (0, 2),
            (0, 7),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 5),
            (1, 4),
            (1, 3),
            (2, 4),
            (2, 8),
            (3, 5),
            (3, 7),
            (4, 7),
            (4, 5),
            (4, 8),
            (4, 6),
            (5, 7),
            (6, 7),
            (7, 8),
        ],
        "start_id": 0,
        "goal_id": 8,
        "max_path_length": 5,
    },
    "10v_25e": {
        "num_nodes": 10,
        "edge_pairs": [
            (0, 5),
            (0, 7),
            (0, 8),
            (1, 2),
            (1, 4),
            (1, 3),
            (1, 5),
            (1, 8),
            (1, 9),
            (2, 5),
            (2, 3),
            (2, 7),
            (2, 4),
            (2, 9),
            (3, 4),
            (3, 9),
            (3, 8),
            (4, 9),
            (4, 8),
            (4, 5),
            (5, 9),
            (6, 9),
            (6, 8),
            (7, 8),
            (7, 9),
        ],
        "start_id": 0,
        "goal_id": 9,
        "max_path_length": 5,
    },
}

BLOCKING_PERCENTAGES = [0, 25, 50, 75, 100]


# =============================================================================
# Helper Functions
# =============================================================================


def find_random_path(
    num_nodes: int,
    edge_pairs: List[Tuple[int, int]],
    start_id: int,
    goal_id: int,
    max_length: int,
    seed: int,
    mandatory_blocked_indices: Set[int] = None,
) -> Tuple[List[int], Set[int]]:
    """
    Find a random path from start to goal using DFS with shuffled neighbors.

    Returns:
        Tuple of (path_as_node_ids, path_edge_indices)
    """
    random.seed(seed)

    # Build adjacency list: node_id -> list of (neighbor_id, edge_index)
    adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_nodes)}
    for idx, (v1, v2) in enumerate(edge_pairs):
        if mandatory_blocked_indices and idx in mandatory_blocked_indices:
            continue
        adj[v1].append((v2, idx))
        adj[v2].append((v1, idx))

    # Collect all valid paths using DFS
    all_paths: List[Tuple[List[int], Set[int]]] = []

    def dfs(
        current: int,
        visited_nodes: Set[int],
        path_nodes: List[int],
        path_edges: Set[int],
    ):
        if current == goal_id:
            all_paths.append((list(path_nodes), set(path_edges)))
            return

        if len(path_edges) >= max_length:
            return

        neighbors = list(adj[current])
        random.shuffle(neighbors)

        for neighbor, edge_idx in neighbors:
            if neighbor in visited_nodes:
                continue

            visited_nodes.add(neighbor)
            path_nodes.append(neighbor)
            path_edges.add(edge_idx)

            dfs(neighbor, visited_nodes, path_nodes, path_edges)

            path_edges.remove(edge_idx)
            path_nodes.pop()
            visited_nodes.remove(neighbor)

    dfs(start_id, {start_id}, [start_id], set())

    if not all_paths:
        raise ValueError(f"No path found from {start_id} to {goal_id}")

    # Pick a random path from all found paths
    return random.choice(all_paths)


def create_edges_with_random_safe_path(
    vertices: List[Vertex],
    edge_pairs: List[Tuple[int, int]],
    start_id: int,
    goal_id: int,
    max_path_length: int,
    block_percent: int,
    seed: int = 42,
    mandatory_blocked_indices: Set[int] = None,
) -> Tuple[List[Edge], List[int], Set[int]]:
    """
    Create edges with a randomly selected safe path and specified blocking percentage.

    1. Find a random valid path from start to goal
    2. Mark those path edges as TRAVERSABLE
    3. Randomly block remaining edges based on block_percent

    Returns:
        Tuple of (edges, safe_path_nodes, safe_path_edge_indices)
    """
    # Find a random path first
    path_nodes, path_edge_indices = find_random_path(
        num_nodes=len(vertices),
        edge_pairs=edge_pairs,
        start_id=start_id,
        goal_id=goal_id,
        max_length=max_path_length,
        seed=seed,
        mandatory_blocked_indices=mandatory_blocked_indices,
    )

    # Now create edges with blocking
    random.seed(seed + 1000)  # Different seed for blocking randomization
    non_path_indices = [i for i in range(len(edge_pairs)) if i not in path_edge_indices]
    
    if mandatory_blocked_indices:
        # These are already blocked, remove them from indices to randomly block
        non_path_indices = [i for i in non_path_indices if i not in mandatory_blocked_indices]

    if block_percent == 100:
        indices_to_block = set(non_path_indices)
    else:
        num_to_block = int(len(non_path_indices) * block_percent / 100)
        indices_to_block = set(random.sample(non_path_indices, num_to_block))

    edges = []
    for i, (v1, v2) in enumerate(edge_pairs):
        if i in path_edge_indices:
            state = EdgeState.TRAVERSABLE
        elif mandatory_blocked_indices and i in mandatory_blocked_indices:
            state = EdgeState.BLOCKED
        elif i in indices_to_block:
            state = EdgeState.BLOCKED
        else:
            state = EdgeState.TRAVERSABLE
        edges.append(Edge(vertices[v1], vertices[v2], state))

    return edges, path_nodes, path_edge_indices

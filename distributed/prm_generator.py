import networkx as nx
import random
from typing import List, Tuple, Dict
import math

from private_path_query_utils import Vertex, Edge, EdgeState, Graph

def generate_prm(num_vertices: int, radius: float, seed: int = 42) -> Graph:
    """
    Generates a guaranteed-connected Probabilistic Roadmap (PRM) style graph using NetworkX's
    random geometric graph algorithm. Iterates seed until a valid graph from start (0) to goal (V-1) is found.
    """
    random.seed(seed)
    
    # Continually iterate seed if the start and goal are disconnected
    while True:
        G = nx.random_geometric_graph(n=num_vertices, radius=radius, seed=seed)
        if num_vertices == 1 or nx.has_path(G, 0, num_vertices - 1):
            break
        seed += 1
        random.seed(seed)
        
    vertices = [Vertex(id=i) for i in range(num_vertices)]
    edges = []
    
    # Turn undirected geometry graphs into directional pairs
    for u, v in G.edges():
        edges.append(Edge(vertices[u], vertices[v], EdgeState.TRAVERSABLE))
        edges.append(Edge(vertices[v], vertices[u], EdgeState.TRAVERSABLE))
        
    return Graph(vertices=vertices, edges=edges)

def partition_edges(graph: Graph, num_parties: int, seed: int = 42) -> Dict[int, List[Edge]]:
    """
    Distributes the graph edges evenly among Bob parties (Bob 1 to num_bobs).
    """
    random.seed(seed)
    num_bobs = num_parties - 1
    
    if num_bobs <= 0:
        return {}
        
    edges = [e for e in graph.edges]
    edges.sort(key=lambda e: (e.vertex1.id, e.vertex2.id))
    
    random.shuffle(edges)
    
    partitions = {b: [] for b in range(1, num_parties)}
    for i, edge in enumerate(edges):
        bob_id = 1 + (i % num_bobs)
        partitions[bob_id].append(edge)
        
    return partitions

def assign_bob_edge_weights(bob_edges: List[Edge], total_budget: float = 1.0, block_fraction: float = 0.5, seed: int = 42) -> Dict[Tuple[int, int], float]:
    """
    Assigns a total weight budget across a subset of Bob's edges to act as "soft-blocks".
    Returns a dictionary mapping directed edge tuples (u, v) to precisely the specific weight constraint.
    """
    random.seed(seed)
    
    # We want to randomly penalize roughly half (or block_fraction) of the edges Bob owns.
    num_to_block = math.ceil(len(bob_edges) * block_fraction)
    if num_to_block == 0 or len(bob_edges) == 0:
        return {}
        
    edges_to_block = random.sample(bob_edges, k=num_to_block)
    
    # Ensure all weights add up exactly to the total_budget limit
    raw_weights = [random.random() for _ in range(num_to_block)]
    total_raw = sum(raw_weights)
    
    edge_weights = {}
    for edge, w in zip(edges_to_block, raw_weights):
        normalized_weight = (w / total_raw) * total_budget
        edge_weights[(edge.vertex1.id, edge.vertex2.id)] = normalized_weight
        
    return edge_weights

"""Prototype geodesic MST, its diameter, and morphological families."""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .exceptions import GraphConstructionError


def pairwise_spherical_distance(X_N: np.ndarray) -> np.ndarray:
    X = np.asarray(X_N, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise GraphConstructionError("Prototype matrix contains zero vectors.")
    X = X / norms
    return np.arccos(np.clip(X @ X.T, -1.0, 1.0))


def build_geodesic_mst(X_N: np.ndarray) -> tuple[nx.Graph, np.ndarray]:
    """Build the source notebook's MST from pairwise spherical geodesic angles."""
    D_geo = pairwise_spherical_distance(X_N)
    if len(D_geo) == 1:
        G = nx.Graph()
        G.add_node(0)
        return G, D_geo

    tree = minimum_spanning_tree(D_geo).tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(len(D_geo)))
    for u, v, distance in zip(tree.row, tree.col, tree.data):
        G.add_edge(
            int(u), int(v),
            geo_distance=float(distance),
        )
    if not nx.is_tree(G):
        raise GraphConstructionError(
            "SciPy MST did not produce a connected tree. Exact duplicate "
            "prototype directions can create zero off-diagonal distances, "
            "which sparse MST treats as absent edges. Resolve duplicates "
            "explicitly or choose a documented tie policy."
        )
    return G, D_geo


def tree_diameter_path(
    G: nx.Graph,
    weight: str = "geo_distance",
) -> list[int]:
    """Weighted two-sweep tree diameter, matching the supplied source."""
    if len(G) == 0:
        return []
    if len(G) == 1:
        return [next(iter(G.nodes))]
    if not nx.is_tree(G):
        raise ValueError("G must be a connected tree.")

    start = next(iter(G.nodes))
    lengths = nx.single_source_dijkstra_path_length(G, start, weight=weight)
    u = max(lengths, key=lengths.get)
    lengths, paths = nx.single_source_dijkstra(G, u, weight=weight)
    v = max(lengths, key=lengths.get)
    return list(paths[v])


def tree_diameter_graph(
    G: nx.Graph,
    weight: str = "geo_distance",
) -> nx.Graph:
    return G.subgraph(tree_diameter_path(G, weight=weight)).copy()


def diam_geometry(
    G: nx.Graph,
    diam: nx.Graph,
    weight: str = "geo_distance",
):
    """Return weighted diameter distance, diameter attachments, and family mapping."""
    if not nx.is_tree(G):
        raise ValueError("G must be a connected tree.")
    diam_nodes = set(diam.nodes)
    if not diam_nodes:
        raise ValueError("diameter must contain at least one node.")
    distance, paths = nx.multi_source_dijkstra(
        G, sources=diam_nodes, weight=weight
    )
    if len(distance) != len(G):
        raise ValueError("Diameter does not reach all graph nodes.")
    attachment = {node: path[0] for node, path in paths.items()}
    family = attachment.copy()
    return distance, attachment, family


def diam_layers(G: nx.Graph, diam: nx.Graph):
    """Assign each graph node to nearest diam node by unweighted edge count."""
    diam_nodes = set(diam.nodes)
    layer, attachment = {}, {}
    queue = deque()
    for node in diam_nodes:
        layer[node] = 0
        attachment[node] = node
        queue.append(node)
    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in layer:
                layer[neighbor] = layer[node] + 1
                attachment[neighbor] = attachment[node]
                queue.append(neighbor)
    return layer, attachment


def family_nodes(
    G: nx.Graph,
    diam_family: dict | None,
    attachment: dict,
) -> dict[int, np.ndarray]:
    families: dict[int, list[int]] = {}
    for node in G.nodes:
        p = attachment[node]
        families.setdefault(p, []).append(node)
    return {p: np.asarray(nodes, dtype=int) for p, nodes in families.items()}


def aggregate_diam_occupancy(
    G: nx.Graph,
    diam: nx.Graph,
    occupancy: np.ndarray,
    diam_attachment: dict | None = None,
    *,
    weight: str = "geo_distance",
):
    occupancy = np.asarray(occupancy, dtype=float)
    if len(occupancy) != len(G):
        raise ValueError("occupancy must have one value per graph node.")
    if diam_attachment is None:
        _, diam_attachment, _ = diam_geometry(G, diam, weight=weight)
    diam_nodes = list(diam.nodes)
    result = {node: 0.0 for node in diam_nodes}
    for node in G.nodes:
        result[diam_attachment[node]] += occupancy[node]
    return (
        np.asarray(diam_nodes, dtype=int),
        np.asarray([result[node] for node in diam_nodes], dtype=float),
    )


def ordered_path_edges(G: nx.Graph) -> list[tuple[int, int]]:
    if len(G) == 0:
        return []
    endpoints = [node for node, degree in G.degree if degree == 1]
    if len(G) == 1:
        return []
    if len(endpoints) != 2:
        raise ValueError("G must be a single undirected path.")
    current, previous = endpoints[0], None
    ordered_edges = []
    while True:
        neighbors = [n for n in G.neighbors(current) if n != previous]
        if not neighbors:
            break
        next_node = neighbors[0]
        ordered_edges.append((current, next_node))
        previous, current = current, next_node
    return ordered_edges


def ordered_path_nodes(G: nx.Graph) -> list[int]:
    edges = ordered_path_edges(G)
    if not edges:
        return list(G.nodes)
    return [edges[0][0]] + [v for _, v in edges]

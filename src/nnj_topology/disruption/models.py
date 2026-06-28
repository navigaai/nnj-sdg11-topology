"""Edge-removal disruption scenarios."""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["random_removal", "targeted_removal", "hazard_removal"]


def _n_to_remove(graph: nx.MultiDiGraph, rho: float) -> int:
    return int(rho * graph.number_of_edges())


def random_removal(graph: nx.MultiDiGraph, rho: float, seed: int) -> nx.MultiDiGraph:
    """Remove a fraction `rho` of edges uniformly at random (percolation-style)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    h = graph.copy()
    k = _n_to_remove(h, rho)
    if k == 0:
        return h
    rng = np.random.default_rng(seed)
    edges = list(h.edges(keys=True))
    idx = rng.choice(len(edges), size=k, replace=False)
    h.remove_edges_from([edges[i] for i in idx])
    return h


def targeted_removal(graph: nx.MultiDiGraph, rho: float, seed: int = 0) -> nx.MultiDiGraph:
    """Remove the top fraction `rho` of edges by edge betweenness (deterministic)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    h = graph.copy()
    k = _n_to_remove(h, rho)
    if k == 0:
        return h
    simple = nx.Graph(h)
    bc = nx.edge_betweenness_centrality(simple, weight="length")
    ranked = sorted(bc, key=lambda e: bc[e], reverse=True)[:k]
    for u, v in ranked:
        for key in list(h.get_edge_data(u, v, default={}).keys()):
            h.remove_edge(u, v, key)
        if h.has_edge(v, u):
            for key in list(h.get_edge_data(v, u, default={}).keys()):
                h.remove_edge(v, u, key)
    return h


def hazard_removal(
    graph: nx.MultiDiGraph, rho: float, seed: int = 0, *, hazard_nodes: set
) -> nx.MultiDiGraph:
    """Remove edges incident to hazard nodes, capped at a fraction `rho` of edges."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    h = graph.copy()
    cap = _n_to_remove(h, rho)
    if cap == 0:
        return h
    incident = [
        (u, v, key)
        for u, v, key in h.edges(keys=True)
        if u in hazard_nodes or v in hazard_nodes
    ]
    rng = np.random.default_rng(seed)
    if len(incident) > cap:
        idx = rng.choice(len(incident), size=cap, replace=False)
        incident = [incident[i] for i in idx]
    h.remove_edges_from(incident)
    return h

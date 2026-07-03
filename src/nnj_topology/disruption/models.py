"""Edge-removal disruption scenarios."""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["random_removal", "targeted_removal", "hazard_removal", "betweenness_ranking"]


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


def betweenness_ranking(graph: nx.MultiDiGraph) -> list:
    """Return undirected edges sorted by edge betweenness centrality, highest first.

    Callers can precompute this once and pass it to repeated ``targeted_removal``
    calls at different rho values to avoid recomputing betweenness every time.
    """
    simple = nx.Graph(graph)
    bc = nx.edge_betweenness_centrality(simple, weight="length")
    return sorted(bc, key=lambda e: bc[e], reverse=True)


def targeted_removal(
    graph: nx.MultiDiGraph, rho: float, seed: int = 0, ranking: list | None = None
) -> nx.MultiDiGraph:
    """Remove the top fraction `rho` of edges by edge betweenness (deterministic).

    `rho` is measured against the directed edge count, matching `random_removal`,
    so the same rho removes the same fraction of directed edges across scenarios.

    Parameters
    ----------
    ranking:
        Optional precomputed edge ranking (list of (u, v) tuples, high→low
        betweenness).  When provided the expensive betweenness computation is
        skipped; results are identical to computing it fresh on the same graph.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    h = graph.copy()
    k = _n_to_remove(h, rho)
    if k == 0:
        return h
    if ranking is None:
        ranking = betweenness_ranking(graph)
    ranked = ranking
    removed = 0
    for u, v in ranked:
        if removed >= k:
            break
        for a, b in ((u, v), (v, u)):
            if removed >= k:
                break
            if h.has_edge(a, b):
                for key in list(h.get_edge_data(a, b).keys()):
                    if removed >= k:
                        break
                    h.remove_edge(a, b, key)
                    removed += 1
    return h


def hazard_removal(
    graph: nx.MultiDiGraph, rho: float, seed: int = 0, *, hazard_nodes: set[int]
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

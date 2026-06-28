"""OpenStreetMap walk-network acquisition and cleanup."""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import osmnx as ox

logger = logging.getLogger(__name__)

__all__ = ["load_walk_network", "largest_connected_component"]


def load_walk_network(
    place: str, crs: str, cache_path: Path | None = None
) -> nx.MultiDiGraph:
    """Load the pedestrian network for `place`, projected to metric `crs`.

    Uses a GraphML cache when `cache_path` exists to avoid re-downloading.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached walk network from %s", cache_path)
        return ox.load_graphml(cache_path)

    logger.info("Downloading walk network for %s", place)
    graph = ox.graph_from_place(place, network_type="walk")
    graph = ox.project_graph(graph, to_crs=crs)
    graph = largest_connected_component(graph)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, cache_path)
        logger.info("Cached walk network to %s", cache_path)
    return graph


def largest_connected_component(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the subgraph induced by the largest weakly connected component."""
    if graph.number_of_nodes() == 0:
        return graph
    components = nx.weakly_connected_components(graph)
    largest = max(components, key=len)
    return graph.subgraph(largest).copy()

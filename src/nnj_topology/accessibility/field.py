"""Network walk-time accessibility field to nearest green space."""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)

__all__ = ["WALK_SPEED_M_PER_MIN", "add_travel_time", "accessibility_field"]

WALK_SPEED_M_PER_MIN: float = 80.0  # ~4.8 km/h


def add_travel_time(
    graph: nx.MultiDiGraph, speed_m_per_min: float = WALK_SPEED_M_PER_MIN
) -> nx.MultiDiGraph:
    """Add a `travel_time` (minutes) edge attribute derived from `length` (metres)."""
    for _, _, data in graph.edges(data=True):
        data["travel_time"] = float(data["length"]) / speed_m_per_min
    return graph


def accessibility_field(
    graph: nx.MultiDiGraph, source_nodes: list[int]
) -> dict[int, float]:
    """Walk-minutes from every node to the nearest green-space access node.

    Multi-source Dijkstra over `travel_time`. Unreachable nodes map to inf.
    """
    if not source_nodes:
        raise ValueError("source_nodes must be non-empty")
    lengths = nx.multi_source_dijkstra_path_length(
        graph, sources=set(source_nodes), weight="travel_time"
    )
    return {node: lengths.get(node, float("inf")) for node in graph.nodes}

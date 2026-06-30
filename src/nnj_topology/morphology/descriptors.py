"""Urban-morphology descriptors (the architecture-mathematics bridge)."""
from __future__ import annotations

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox

logger = logging.getLogger(__name__)

__all__ = ["morphology_descriptors", "greenspace_fragmentation"]


def morphology_descriptors(graph: nx.MultiDiGraph) -> dict[str, float]:
    """Compute intersection density, circuity, orientation entropy, mean block size."""
    # Ensure street_count attributes are set (required by osmnx 2.1.0)
    street_counts = ox.stats.count_streets_per_node(graph)
    for node, count in street_counts.items():
        graph.nodes[node]["street_count"] = count

    stats = ox.stats.basic_stats(graph)

    # For orientation entropy, we need an unprojected graph (lat/lon coordinates)
    # If the graph is projected, unproject it temporarily
    graph_for_bearing = graph
    if ox.projection.is_projected(graph.graph.get("crs")):
        graph_for_bearing = ox.projection.project_graph(graph, to_crs="EPSG:4326")

    graph_b = ox.bearing.add_edge_bearings(graph_for_bearing)
    entropy = float(ox.bearing.orientation_entropy(graph_b))

    return {
        "intersection_density": float(stats.get("intersection_count", 0))
        / max(float(stats.get("edge_length_total", 1.0)) / 1000.0, 1e-9),
        "circuity": float(stats.get("circuity_avg", float("nan"))),
        "orientation_entropy": entropy,
        "mean_block_size": float(stats.get("street_length_avg", float("nan"))),
    }


def greenspace_fragmentation(green: gpd.GeoDataFrame) -> float:
    """Patch count per square kilometre of green area (higher = more fragmented)."""
    if len(green) == 0:
        return 0.0
    total_area_km2 = float(green.geometry.area.sum()) / 1e6
    if total_area_km2 <= 0:
        return 0.0
    return len(green) / total_area_km2

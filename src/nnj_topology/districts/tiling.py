"""H3 district tiling and per-district local resilience."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List

import geopandas as gpd
import h3
import networkx as nx
from shapely.geometry import Point

from nnj_topology.disruption.resilience import ResilienceResult, resilience_curve
from nnj_topology.topology.diagrams import Diagram, sublevel_diagram
from nnj_topology.topology.distances import wasserstein_distance

logger = logging.getLogger(__name__)

__all__ = ["assign_nodes_to_hexes", "local_diagram", "district_resilience"]


def assign_nodes_to_hexes(
    graph: nx.MultiDiGraph, crs: str, h3_res: int
) -> Dict[str, List]:
    """Map each H3 cell id to the node ids whose coordinates fall inside it."""
    node_ids = list(graph.nodes)
    pts = gpd.GeoSeries(
        [Point(float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in node_ids],
        crs=crs,
    ).to_crs("EPSG:4326")
    mapping: Dict[str, List] = {}
    for node, geom in zip(node_ids, pts):
        cell = h3.latlng_to_cell(geom.y, geom.x, h3_res)
        mapping.setdefault(cell, []).append(node)
    return mapping


def local_diagram(
    graph: nx.Graph, field: dict, node_ids: List, max_dim: int = 1
) -> Diagram:
    """Sublevel persistence on the subgraph induced by `node_ids`."""
    sub = graph.subgraph(node_ids)
    sub_field = {n: field[n] for n in node_ids}
    return sublevel_diagram(sub, sub_field, max_dim=max_dim)


def district_resilience(
    graph: nx.MultiDiGraph,
    field_fn: Callable[[nx.MultiDiGraph], dict],
    disrupt: Callable[..., nx.MultiDiGraph],
    rhos: List[float],
    n_replicates: int,
    seed: int,
    hex_nodes: Dict[str, List],
    max_dim: int = 1,
    min_nodes: int = 10,
    dim: int = 0,
    min_persistence: float = 0.0,
) -> Dict[str, ResilienceResult]:
    """Per-hex resilience curve, computed by streaming over disruptions.

    Each disrupted graph is built exactly once per ``(rho, replicate)`` and its
    per-hex distance contributions are accumulated immediately, so only ONE
    disrupted graph + field is resident in memory at a time (avoids holding all
    ``len(rhos) * n_replicates`` full-city copies simultaneously). Results are
    identical to averaging the replicates directly.

    `field_fn(graph) -> dict` recomputes the accessibility field on a (possibly
    disrupted) graph. Hexes with fewer than `min_nodes` nodes are skipped.
    """
    base_field = field_fn(graph)
    simple = nx.Graph(graph)
    qualifying = [cell for cell, nodes in hex_nodes.items() if len(nodes) >= min_nodes]
    base_local = {
        cell: local_diagram(simple, base_field, hex_nodes[cell], max_dim)
        for cell in qualifying
    }

    nonzero_rhos = [rho for rho in rhos if rho != 0.0]
    # Per-hex running sum of Wasserstein distances, keyed by rho.
    dist_sum: Dict[str, Dict[float, float]] = {
        cell: {rho: 0.0 for rho in nonzero_rhos} for cell in qualifying
    }

    # Stream: one disrupted graph resident at a time; score every hex, then drop it.
    for rho in nonzero_rhos:
        for rep in range(n_replicates):
            dg = disrupt(graph, rho, seed=seed + rep)
            simple_dg = nx.Graph(dg)
            dg_field = field_fn(dg)
            for cell in qualifying:
                local = local_diagram(simple_dg, dg_field, hex_nodes[cell], max_dim)
                dist_sum[cell][rho] += wasserstein_distance(
                    base_local[cell], local, dim=dim, min_persistence=min_persistence
                )
            del dg, simple_dg, dg_field  # release before the next (rho, rep)

    denom = float(n_replicates) if n_replicates else 1.0
    results: Dict[str, ResilienceResult] = {}
    for cell in qualifying:
        means = {rho: dist_sum[cell][rho] / denom for rho in nonzero_rhos}

        def distance_at_rho(rho: float, _means=means) -> float:
            return 0.0 if rho == 0.0 else float(_means[rho])

        results[cell] = resilience_curve(rhos, distance_at_rho)
    return results

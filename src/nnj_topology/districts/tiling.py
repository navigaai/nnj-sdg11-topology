"""H3 district tiling and per-district local resilience."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
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
) -> Dict[str, ResilienceResult]:
    """Per-hex resilience curve; disrupted graphs are computed once and reused.

    `field_fn(graph) -> dict` recomputes the accessibility field on a (possibly
    disrupted) graph. Hexes with fewer than `min_nodes` nodes are skipped.
    """
    base_field = field_fn(graph)
    simple = nx.Graph(graph)
    base_local = {
        cell: local_diagram(simple, base_field, nodes, max_dim)
        for cell, nodes in hex_nodes.items()
        if len(nodes) >= min_nodes
    }

    # Pre-compute disrupted graphs + their fields + simple views, once per (rho, rep).
    disrupted: Dict[float, list] = {}
    for rho in rhos:
        if rho == 0.0:
            continue
        reps = []
        for rep in range(n_replicates):
            dg = disrupt(graph, rho, seed=seed + rep)
            reps.append((nx.Graph(dg), field_fn(dg)))
        disrupted[rho] = reps

    results: Dict[str, ResilienceResult] = {}
    for cell, nodes in hex_nodes.items():
        if cell not in base_local:
            continue
        base_dgm = base_local[cell]

        def distance_at_rho(rho: float, _nodes=nodes, _base=base_dgm) -> float:
            if rho == 0.0:
                return 0.0
            dists = []
            for simple_dg, dg_field in disrupted[rho]:
                local = local_diagram(simple_dg, dg_field, _nodes, max_dim)
                dists.append(wasserstein_distance(_base, local, dim=1))
            return float(np.mean(dists))

        results[cell] = resilience_curve(rhos, distance_at_rho)
    return results

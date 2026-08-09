"""Walking-speed sensitivity (reviewer A).

Walking speed enters only as travel_time = length / speed, so it is a GLOBAL linear
rescaling of the accessibility field. The sublevel diagram, Wasserstein distance and
AUC all scale by the same constant, and regression signs, significance and the city
ranking are therefore invariant to the speed choice (a proposition). If the
persistence threshold is scaled with speed, the invariance is exact; with the
threshold held fixed at 1 walk-minute a small nonlinearity remains. This script
confirms empirically on Amsterdam that district AUC at 4.0 / 5.0 km/h is a near-exact
rescaling of the 4.8 km/h headline (Spearman ~ 1).

Usage: uv run python scripts/speed_sensitivity.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
from scipy.stats import spearmanr

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import access_points, load_greenspace, snap_points_to_nodes
from nnj_topology.data.network import load_walk_network
from nnj_topology.districts.tiling import assign_nodes_to_hexes, district_resilience
from nnj_topology.disruption.models import random_removal

logging.getLogger().setLevel(logging.WARNING)
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]


def auc_at_speed(g, src, hn, kmh: float) -> dict:
    g = add_travel_time(nx.MultiDiGraph(g), speed_m_per_min=kmh * 1000 / 60)
    res = district_resilience(
        g, lambda gg: accessibility_field(gg, src), random_removal,
        rhos=RHOS, n_replicates=2, seed=42, hex_nodes=hn,
        max_dim=0, min_nodes=10, dim=0, min_persistence=1.0)
    return {c: r.auc for c, r in res.items()}


def main() -> None:
    crs = "EPSG:28992"
    g = load_walk_network("Amsterdam, Netherlands", crs, Path("data/amsterdam/walk.graphml"))
    g = clip_graph_to_boundary(g, load_urban_boundary(
        Path("data/ghsl/ghs_ucdb.gpkg"), "amsterdam", crs), crs)
    green = load_greenspace("Amsterdam, Netherlands", crs, Path("data/amsterdam/green.gpkg"))
    src = snap_points_to_nodes(access_points(green), add_travel_time(g))
    hn = assign_nodes_to_hexes(g, crs, 8)

    ref = auc_at_speed(g, src, hn, 4.8)
    for kmh in (4.0, 5.0):
        a = auc_at_speed(g, src, hn, kmh)
        cells = [c for c in ref if c in a]
        rho = spearmanr([ref[c] for c in cells], [a[c] for c in cells]).correlation
        ratios = [a[c] / ref[c] for c in cells if ref[c] > 0]
        med = sorted(ratios)[len(ratios) // 2]
        print(f"{kmh} km/h vs 4.8: Spearman={rho:.4f}, median AUC ratio={med:.3f} "
              f"(expected ~{4.8 / kmh:.3f})")


if __name__ == "__main__":
    main()

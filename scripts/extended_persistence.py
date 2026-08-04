"""Extended-persistence (finite-cap) variant to fix the disconnection artefact
(reviewer construct-validity point).

The benchmark (scripts/benchmark_validation.py) exposed that the headline metric
correlates with the WRONG sign against connectivity degradation, because nodes that
become unreachable acquire an infinite field value and drop out of the finite
persistence diagram. Here we test a fix: cap the field at a large finite value
TAU (60 walk-minutes = "effectively unreachable") instead of infinity, so
disconnected nodes remain in the diagram as high-birth H0 classes and disruption
that disconnects them *increases* the Wasserstein distance (correct direction).
This is a finite-cap stand-in for extended persistence.

For an apples-to-apples test we recompute BOTH the original (inf) and the capped
metric on the SAME reduced disruption grid, then compare their Spearman correlation
with the connectivity-degradation baseline. If the cap fixes the artefact, the
capped metric's correlation flips from negative to positive.

Writes output/extended_persistence.csv. Usage: uv run python scripts/extended_persistence.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import access_points, load_greenspace, snap_points_to_nodes
from nnj_topology.data.network import load_walk_network
from nnj_topology.disruption.models import random_removal
from nnj_topology.districts.tiling import assign_nodes_to_hexes, district_resilience

logging.getLogger().setLevel(logging.WARNING)

CITIES = {
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
    "bogota": ("Bogotá, Colombia", "EPSG:32618"),
}
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_REP = 2
TAU = 60.0  # walk-minutes; cap for "effectively unreachable" nodes


def city_metrics(city: str, place: str, crs: str) -> pd.DataFrame:
    g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
    g = clip_graph_to_boundary(g, load_urban_boundary(
        Path("data/ghsl/ghs_ucdb.gpkg"), city, crs), crs)
    g = add_travel_time(g)
    green = load_greenspace(place, crs, Path(f"data/{city}/green.gpkg"))
    src = snap_points_to_nodes(access_points(green), g)
    hn = assign_nodes_to_hexes(g, crs, 8)

    def field_inf(gg):
        return accessibility_field(gg, src)

    def field_cap(gg):
        return {n: (v if np.isfinite(v) else TAU)
                for n, v in accessibility_field(gg, src).items()}

    kw = dict(rhos=RHOS, n_replicates=N_REP, seed=42, hex_nodes=hn,
              max_dim=0, min_nodes=10, dim=0, min_persistence=1.0)
    res_inf = district_resilience(g, field_inf, random_removal, **kw)
    res_cap = district_resilience(g, field_cap, random_removal, **kw)

    rows = []
    for c in res_inf:
        if c in res_cap:
            rows.append({"city": city, "hex": c,
                         "auc_inf": res_inf[c].auc, "auc_cap": res_cap[c].auc})
    print(f"  {city}: {len(rows)} districts")
    return pd.DataFrame(rows)


def main() -> None:
    met = pd.concat([city_metrics(c, p, crs) for c, (p, crs) in CITIES.items()],
                    ignore_index=True)
    met.to_csv("output/extended_persistence.csv", index=False)

    bench = pd.read_csv("output/benchmark_validation.csv")
    df = met.merge(bench, on=["city", "hex"], how="inner")
    print(f"\nmerged n = {len(df)}")

    print("\n=== Convergent validity vs the two baselines (reduced grid) ===")
    print(f"{'metric':12s} {'vs connectivity-degrad':>24s} {'vs distance-degrad':>22s}")
    for col, label in [("auc_inf", "original (inf)"), ("auc_cap", "capped (TAU=60)")]:
        rc = spearmanr(df[col], df["degrad_unreach_auc"]).correlation
        rd = spearmanr(df[col], df["degrad_access_auc"]).correlation
        print(f"{label:14s} {rc:+.3f}{'':18s} {rd:+.3f}")
    print("\n(If the cap fixes the artefact, capped vs connectivity-degrad flips "
          "from negative to positive.)")


if __name__ == "__main__":
    main()

"""Benchmark / convergent validation of the topological resilience metric
(reviewer major points M1 and M2).

Two non-topological baselines are computed per district under the SAME random
disruption, on a reduced grid (cheap: no persistent homology, no Wasserstein):

  M1 (distance-based degradation): the increase in mean network walk-time to the
     nearest green space over the district's nodes, integrated over rho.
     -> "does the topological metric add anything over a simple accessibility-
        degradation measure?"

  M2 (connectivity degradation): the increase in the fraction of district nodes
     that become unreachable to any green space (finite -> infinite field),
     integrated over rho. H0 persistence of a sublevel filtration is single-linkage
     connectivity, so this is the natural non-persistent connectivity baseline.
     -> "does the H0-Wasserstein machinery add anything over plain connectivity?"

We then report Spearman correlations between the topological district AUC and each
baseline, and refit the morphology regression controlling for the distance-based
baseline (does morphology still predict the TOPOLOGICAL metric beyond the simple
one?).

Writes output/benchmark_validation.csv (+ prints). Reduced grid keeps this cheap.
Usage: uv run python scripts/benchmark_validation.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import access_points, load_greenspace, snap_points_to_nodes
from nnj_topology.data.network import load_walk_network
from nnj_topology.disruption.models import random_removal
from nnj_topology.districts.tiling import assign_nodes_to_hexes

logging.getLogger().setLevel(logging.WARNING)

CITIES = {
    "istanbul": ("İstanbul, Turkey", "EPSG:32635"),
    "barcelona": ("Barcelona, Spain", "EPSG:25831"),
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
    "bogota": ("Bogotá, Colombia", "EPSG:32618"),
    "phoenix": ("Phoenix, Arizona, USA", "EPSG:26912"),
}
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_REP = 3
CAP = 60.0  # walk-minutes cap for unreachable nodes when averaging degradation


def _hex_scalars(field: dict, nodes: list) -> tuple[float, float]:
    """(mean finite walk-time, fraction unreachable) over a district's nodes."""
    vals = np.array([field.get(n, np.inf) for n in nodes])
    finite = np.isfinite(vals)
    mean_acc = float(np.mean(np.clip(vals[finite], 0, CAP))) if finite.any() else CAP
    frac_inf = float(np.mean(~finite))
    return mean_acc, frac_inf


def city_baselines(city: str, place: str, crs: str) -> pd.DataFrame:
    graph = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
    boundary = load_urban_boundary(Path("data/ghsl/ghs_ucdb.gpkg"), city, crs)
    graph = clip_graph_to_boundary(graph, boundary, crs)
    graph = add_travel_time(graph)
    green = load_greenspace(place, crs, Path(f"data/{city}/green.gpkg"))
    sources = snap_points_to_nodes(access_points(green), graph)
    hex_nodes = assign_nodes_to_hexes(graph, crs, 8)

    base_field = accessibility_field(graph, sources)
    base = {c: _hex_scalars(base_field, ns) for c, ns in hex_nodes.items()}

    # accumulate degradation curves per hex
    deg_acc = {c: [] for c in hex_nodes}   # mean-access increase per rho
    deg_inf = {c: [] for c in hex_nodes}   # unreachable-fraction increase per rho
    for rho in RHOS:
        acc_r = {c: [] for c in hex_nodes}
        inf_r = {c: [] for c in hex_nodes}
        reps = 1 if rho == 0.0 else N_REP
        for rep in range(reps):
            g = graph if rho == 0.0 else random_removal(graph, rho, seed=rep)
            fld = base_field if rho == 0.0 else accessibility_field(g, sources)
            for c, ns in hex_nodes.items():
                ma, fi = _hex_scalars(fld, ns)
                acc_r[c].append(ma)
                inf_r[c].append(fi)
        for c in hex_nodes:
            deg_acc[c].append(np.mean(acc_r[c]) - base[c][0])
            deg_inf[c].append(np.mean(inf_r[c]) - base[c][1])

    rows = []
    for c, ns in hex_nodes.items():
        if graph.subgraph(ns).number_of_edges() == 0:
            continue
        rows.append({
            "city": city, "hex": c,
            "base_access_min": base[c][0],
            "degrad_access_auc": float(np.trapezoid(deg_acc[c], RHOS) / (RHOS[-1] - RHOS[0])),
            "degrad_unreach_auc": float(np.trapezoid(deg_inf[c], RHOS) / (RHOS[-1] - RHOS[0])),
        })
    print(f"  {city}: {len(rows)} districts")
    return pd.DataFrame(rows)


def main() -> None:
    base = pd.concat([city_baselines(c, p, crs) for c, (p, crs) in CITIES.items()],
                     ignore_index=True)
    base.to_csv("output/benchmark_validation.csv", index=False)

    df = pd.read_csv("output/district_table.csv").merge(base, on=["city", "hex"], how="inner")
    print(f"\nmerged n = {len(df)}")

    print("\n=== Convergent validity: Spearman(topological AUC, baseline) ===")
    for col, label in [("degrad_access_auc", "M1 distance-degradation"),
                       ("degrad_unreach_auc", "M2 connectivity-degradation"),
                       ("base_access_min", "static baseline access")]:
        rho, p = spearmanr(df["auc"], df[col])
        print(f"  {label:32s}: rho={rho:+.3f} (p={p:.2g})")

    print("\n=== Added value: does topological AUC keep the morphology signal "
          "AFTER controlling for the simple baselines? ===")
    m = smf.ols(
        "auc ~ circuity + mean_street_length + orientation_entropy + "
        "intersection_density + greenspace_fragmentation + "
        "degrad_access_auc + degrad_unreach_auc + C(city)", data=df).fit()
    for t in ["circuity", "mean_street_length", "orientation_entropy",
              "degrad_access_auc", "degrad_unreach_auc"]:
        print(f"  {t:22s}: coef={m.params[t]:+.4g}  p={m.pvalues[t]:.2g}")
    print(f"  R2 = {m.rsquared:.3f}")


if __name__ == "__main__":
    main()

"""Mechanism test: does network path redundancy mediate the morphology -> resilience
association? (reviewer: the paper argues resilience arises because alternative paths
survive, but never measures redundancy as a variable.)

An independent per-district redundancy measure is computed from graph STRUCTURE alone
(no disruption simulation, no accessibility field, so it is not circular with the
topological AUC outcome):

  meshedness (alpha index): density of independent cycles in the district subgraph,
    alpha = (e - v + p) / (2v - 5), the standard planar-network redundancy / route-
    availability measure (Buhl et al. 2006; Cardillo et al. 2006). More independent
    cycles = more alternative ways between locations = more path redundancy.

  avg_degree: mean undirected node degree, a coarse connectivity/redundancy proxy.

Mediation (Baron-Kenny style, with city fixed effects throughout, since the headline
effect is identified within city):
  (a) resilience AUC ~ morphology + FE                    [total effect, known]
  (b) redundancy    ~ morphology + FE                     [morphology -> mediator]
  (c) AUC ~ morphology + redundancy + FE                  [mediator -> outcome, and
                                                           attenuation of morphology]

Reports within-city Spearman(redundancy, AUC), and the % attenuation of the circuity
and mean-street-length coefficients when redundancy is added.

Reads output/district_table.csv + cached graphs. Writes output/redundancy_mediation.csv.
Usage: uv run python scripts/redundancy_mediation.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.network import load_walk_network
from nnj_topology.districts.tiling import assign_nodes_to_hexes

logging.getLogger().setLevel(logging.WARNING)

CITIES = {
    "istanbul": ("İstanbul, Turkey", "EPSG:32635"),
    "barcelona": ("Barcelona, Spain", "EPSG:25831"),
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
    "bogota": ("Bogotá, Colombia", "EPSG:32618"),
    "phoenix": ("Phoenix, Arizona, USA", "EPSG:26912"),
    "singapore": ("Singapore", "EPSG:32648"),
    "nairobi": ("Nairobi, Kenya", "EPSG:32737"),
    "vienna": ("Vienna, Austria", "EPSG:32633"),
}
MORPH = ["circuity", "mean_street_length", "orientation_entropy",
         "intersection_density", "greenspace_fragmentation"]
HEAD = ["circuity", "mean_street_length", "orientation_entropy"]


def district_redundancy(city: str, place: str, crs: str) -> pd.DataFrame:
    g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
    g = clip_graph_to_boundary(g, load_urban_boundary(
        Path("data/ghsl/ghs_ucdb.gpkg"), city, crs), crs)
    hex_nodes = assign_nodes_to_hexes(g, crs, 8)
    rows = []
    for cell, ns in hex_nodes.items():
        sub = g.subgraph(ns)
        u = sub.to_undirected()  # collapse parallel/reciprocal edges for cycle count
        v = u.number_of_nodes()
        e = u.number_of_edges()
        if v < 3 or e == 0:
            continue
        import networkx as nx
        p = nx.number_connected_components(u)
        denom = 2 * v - 5
        alpha = (e - v + p) / denom if denom > 0 else np.nan
        rows.append({"city": city, "hex": cell,
                     "meshedness": float(alpha),
                     "avg_degree": float(2 * e / v)})
    print(f"  {city}: {len(rows)} districts")
    return pd.DataFrame(rows)


def _report_reg(df, extra=None):
    rhs = " + ".join(MORPH + ([extra] if extra else []) + ["C(city)"])
    return smf.ols(f"auc ~ {rhs}", data=df).fit()


def main() -> None:
    red = pd.concat([district_redundancy(c, p, crs) for c, (p, crs) in CITIES.items()],
                    ignore_index=True)
    red.to_csv("output/redundancy_mediation.csv", index=False)
    df = pd.read_csv("output/district_table.csv").merge(red, on=["city", "hex"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["meshedness", "avg_degree"])
    print(f"\nmerged n = {len(df)}")

    print("\n=== (0) Within-city Spearman(redundancy, resilience AUC) ===")
    for col in ["meshedness", "avg_degree"]:
        for c in [col]:
            df[c + "_wr"] = df.groupby("city")[c].rank()
        df["auc_wr"] = df.groupby("city")["auc"].rank()
        r, p = spearmanr(df[col + "_wr"], df["auc_wr"])
        print(f"  {col:12s}: within-city pooled rho={r:+.3f} (p={p:.2g})")

    print("\n=== (a) morphology -> redundancy (meshedness ~ morphology + FE) ===")
    mb = smf.ols("meshedness ~ " + " + ".join(MORPH + ["C(city)"]), data=df).fit()
    for t in HEAD:
        print(f"  {t:20s}: coef={mb.params[t]:+.4g}  p={mb.pvalues[t]:.2g}")

    print("\n=== (b,c) mediation: AUC coefficients before/after adding meshedness ===")
    m0 = _report_reg(df)                    # total effect
    m1 = _report_reg(df, "meshedness")      # controlling mediator
    print(f"  {'term':20s} {'total':>10s} {'+mesh':>10s} {'attenuation':>12s}")
    for t in HEAD:
        b0, b1 = m0.params[t], m1.params[t]
        att = 100 * (b0 - b1) / b0 if b0 != 0 else np.nan
        print(f"  {t:20s} {b0:>+10.4g} {b1:>+10.4g} {att:>11.1f}%   "
              f"(p {m0.pvalues[t]:.1g}->{m1.pvalues[t]:.1g})")
    print(f"  meshedness coef in (c): {m1.params['meshedness']:+.4g} "
          f"(p={m1.pvalues['meshedness']:.2g})")
    print(f"  R2: total={m0.rsquared:.3f}  +meshedness={m1.rsquared:.3f}")


if __name__ == "__main__":
    main()

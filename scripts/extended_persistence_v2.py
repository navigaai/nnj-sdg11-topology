"""Principled extended-persistence resilience metric + criterion validation
(reviewer W1 + W2).

W2: the naive finite cap fixed the disconnection sign but swamped the graded signal
because all capped nodes collapsed into ordinary H0. Extended persistence
(scripts/_extdiag.py) instead routes disconnection into a SEPARATE extended
subdiagram, so both channels are kept. We compute three district metrics on the same
reduced random-disruption grid:
    auc_inf : original (inf, drops disconnected)  -- headline metric
    auc_cap : naive finite cap                    -- previous diagnostic
    auc_ext : extended persistence (the fix)
and correlate each with the two non-topological baselines from
benchmark_validation.csv.

W1: the distance-degradation baseline is exactly a proximity-accessibility
degradation (an established-style measure). A STRONG POSITIVE correlation of auc_ext
with it is the criterion validity the metric was missing.

Writes output/extended_persistence_v2.csv. Usage:
    uv run python scripts/extended_persistence_v2.py [city1 city2 ...]
(default: amsterdam bogota)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from persim import wasserstein
from scipy.stats import spearmanr

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import access_points, load_greenspace, snap_points_to_nodes
from nnj_topology.data.network import load_walk_network
from nnj_topology.disruption.models import random_removal
from nnj_topology.districts.tiling import assign_nodes_to_hexes, local_diagram

sys.path.insert(0, str(Path(__file__).parent))
from _extdiag import extended_h0  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

CITY_META = {
    "istanbul": ("İstanbul, Turkey", "EPSG:32635"),
    "barcelona": ("Barcelona, Spain", "EPSG:25831"),
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
    "bogota": ("Bogotá, Colombia", "EPSG:32618"),
    "phoenix": ("Phoenix, Arizona, USA", "EPSG:26912"),
}
RHOS = [0.1, 0.2, 0.3, 0.4]
N_REP = 2
TAU = 60.0
MINP = 1.0
MIN_NODES = 10


def _finite_h0(sub, field) -> np.ndarray:
    """Original (inf) metric: sublevel H0 finite classes, disconnected dropped."""
    d = local_diagram(sub, field, list(sub.nodes), max_dim=0)
    h0 = d.get(0, np.empty((0, 2)))
    if h0.size == 0:
        return np.empty((0, 2))
    fin = h0[np.isfinite(h0[:, 1])]
    return fin[(fin[:, 1] - fin[:, 0]) >= MINP] if fin.size else np.empty((0, 2))


def _cap_h0(sub, field) -> np.ndarray:
    """Naive-cap metric: replace inf by TAU, ordinary sublevel H0."""
    capped = {n: (v if np.isfinite(v) else TAU) for n, v in field.items()}
    return _finite_h0(sub, capped)


def _auc(dists: dict) -> float:
    xs = [0.0] + RHOS
    ys = [0.0] + [dists[r] for r in RHOS]
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def city_metrics(city: str, place: str, crs: str) -> pd.DataFrame:
    g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
    g = clip_graph_to_boundary(g, load_urban_boundary(
        Path("data/ghsl/ghs_ucdb.gpkg"), city, crs), crs)
    g = add_travel_time(g)
    green = load_greenspace(place, crs, Path(f"data/{city}/green.gpkg"))
    src = snap_points_to_nodes(access_points(green), g)
    hn = assign_nodes_to_hexes(g, crs, 8)
    cells = [c for c, ns in hn.items() if len(ns) >= MIN_NODES]

    simple = nx.Graph(g)
    base_field = accessibility_field(g, src)
    base = {}
    for c in cells:
        sub = simple.subgraph(hn[c])
        bf = {n: base_field[n] for n in hn[c]}
        base[c] = (_finite_h0(sub, bf), _cap_h0(sub, bf),
                   extended_h0(sub, bf, TAU, MINP))

    acc = {c: {"inf": {r: 0.0 for r in RHOS}, "cap": {r: 0.0 for r in RHOS},
               "ext": {r: 0.0 for r in RHOS}} for c in cells}
    for r in RHOS:
        for rep in range(N_REP):
            dg = random_removal(g, r, seed=42 + rep)
            sdg = nx.Graph(dg)
            df_ = accessibility_field(dg, src)
            for c in cells:
                ns = [n for n in hn[c] if n in sdg]
                sub = sdg.subgraph(ns)
                ff = {n: df_[n] for n in ns}
                acc[c]["inf"][r] += wasserstein(base[c][0], _finite_h0(sub, ff))
                acc[c]["cap"][r] += wasserstein(base[c][1], _cap_h0(sub, ff))
                acc[c]["ext"][r] += wasserstein(base[c][2], extended_h0(sub, ff, TAU, MINP))
    rows = []
    for c in cells:
        rows.append({
            "city": city, "hex": c,
            "auc_inf": _auc({r: acc[c]["inf"][r] / N_REP for r in RHOS}),
            "auc_cap": _auc({r: acc[c]["cap"][r] / N_REP for r in RHOS}),
            "auc_ext": _auc({r: acc[c]["ext"][r] / N_REP for r in RHOS}),
        })
    print(f"  {city}: {len(rows)} districts")
    return pd.DataFrame(rows)


def main() -> None:
    cities = sys.argv[1:] or ["amsterdam", "bogota"]
    met = pd.concat([city_metrics(c, *CITY_META[c]) for c in cities],
                    ignore_index=True)
    met.to_csv("output/extended_persistence_v2.csv", index=False)
    bench = pd.read_csv("output/benchmark_validation.csv")
    df = met.merge(bench, on=["city", "hex"], how="inner")
    print(f"\nmerged n = {len(df)}  cities={cities}")
    print(f"\n{'metric':16s}{'vs distance-degrad (W1)':>26s}{'vs connectivity-degrad':>24s}")
    for col, lab in [("auc_inf", "original(inf)"), ("auc_cap", "naive-cap"),
                     ("auc_ext", "extended-persist")]:
        rd = spearmanr(df[col], df["degrad_access_auc"]).correlation
        rc = spearmanr(df[col], df["degrad_unreach_auc"]).correlation
        print(f"{lab:16s}{rd:+.3f}{'':20s}{rc:+.3f}")


if __name__ == "__main__":
    main()

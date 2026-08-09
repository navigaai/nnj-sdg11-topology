"""Population-weighted criterion validation (open-source population data).

Reviewers asked for (a) a benchmark against a population-based accessibility measure
and (b) an engagement with the SDG 11.7 population-coverage dimension the network
metric does not directly measure. We use open GHS-POP R2023A (JRC, 1 km, 2020;
same GHSL family and Mollweide grid as our GHS-UCDB boundaries) to build a
POPULATION-WEIGHTED accessibility-degradation benchmark per district: the increase,
under the same random disruption, in the population-weighted mean walk-time to green
space (a 2SFCA-flavoured population-coverage degradation). We then correlate the
topological district AUC with it, per city and within city (the level the city fixed
effects identify from).

Writes output/population_validation.csv. Usage: uv run python scripts/population_validation.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
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
POP = Path("data/ghsl/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif")
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_REP = 2
CAP = 60.0


def node_population(graph, crs) -> dict:
    """Sample GHS-POP (persons per 1 km cell) at each node."""
    with rasterio.open(POP) as src:
        pop_crs = src.crs
        band = src.read(1)
        nod = src.nodata
    nodes = list(graph.nodes)
    xs = np.array([float(graph.nodes[n]["x"]) for n in nodes])
    ys = np.array([float(graph.nodes[n]["y"]) for n in nodes])
    tr = Transformer.from_crs(crs, pop_crs, always_xy=True)
    lon, lat = tr.transform(xs, ys)
    with rasterio.open(POP) as src:
        rc = [src.index(x, y) for x, y in zip(lon, lat)]
    h, w = band.shape
    out = {}
    for n, (r, c) in zip(nodes, rc):
        if 0 <= r < h and 0 <= c < w:
            v = band[r, c]
            out[n] = max(0.0, float(v)) if (nod is None or v != nod) else 0.0
        else:
            out[n] = 0.0
    return out


def _wmean(field, nodes, pop) -> float:
    """Population-weighted mean walk-time over REACHABLE (finite) nodes only.

    Graded access degradation (matches the distance-based benchmark's finite-node
    treatment); disconnection is a separate channel we do not fold in here.
    """
    fin = [n for n in nodes if np.isfinite(field.get(n, np.inf))]
    if not fin:
        return CAP
    vals = np.array([min(field[n], CAP) for n in fin])
    w = np.array([pop.get(n, 0.0) for n in fin])
    if w.sum() <= 0:
        return float(np.mean(vals))
    return float(np.average(vals, weights=w))


def city_pop_degrad(city, place, crs) -> pd.DataFrame:
    g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
    g = clip_graph_to_boundary(g, load_urban_boundary(
        Path("data/ghsl/ghs_ucdb.gpkg"), city, crs), crs)
    g = add_travel_time(g)
    green = load_greenspace(place, crs, Path(f"data/{city}/green.gpkg"))
    src = snap_points_to_nodes(access_points(green), g)
    hn = assign_nodes_to_hexes(g, crs, 8)
    pop = node_population(g, crs)

    base = accessibility_field(g, src)
    base_w = {c: _wmean(base, ns, pop) for c, ns in hn.items()}
    deg = {c: [] for c in hn}
    for rho in RHOS:
        reps = 1 if rho == 0.0 else N_REP
        acc = {c: [] for c in hn}
        for rep in range(reps):
            fld = base if rho == 0.0 else accessibility_field(random_removal(g, rho, seed=rep), src)
            for c, ns in hn.items():
                acc[c].append(_wmean(fld, ns, pop))
        for c in hn:
            deg[c].append(np.mean(acc[c]) - base_w[c])
    rows = []
    for c, ns in hn.items():
        if g.subgraph(ns).number_of_edges() == 0:
            continue
        rows.append({"city": city, "hex": c,
                     "popw_degrad_auc": float(np.trapezoid(deg[c], RHOS) / (RHOS[-1] - RHOS[0])),
                     "pop_total": float(sum(pop.get(n, 0.0) for n in ns))})
    print(f"  {city}: {len(rows)} districts, pop~{sum(pop.values())/1e6:.1f}M")
    return pd.DataFrame(rows)


def main() -> None:
    pv = pd.concat([city_pop_degrad(c, *m) for c, m in CITIES.items()], ignore_index=True)
    pv.to_csv("output/population_validation.csv", index=False)
    df = pd.read_csv("output/district_table.csv").merge(pv, on=["city", "hex"])
    print(f"\nmerged n={len(df)}")
    print("\n=== Criterion validity vs POPULATION-WEIGHTED degradation ===")
    print("per-city Spearman(AUC, popw_degrad):")
    for c, gg in df.groupby("city"):
        r, p = spearmanr(gg.auc, gg.popw_degrad_auc)
        print(f"  {c:10s} {r:+.3f} (p={p:.1e}, n={len(gg)})")
    for col in ["auc", "popw_degrad_auc"]:
        df[col + "_wr"] = df.groupby("city")[col].rank()
    r, p = spearmanr(df.auc_wr, df.popw_degrad_auc_wr)
    rp, pp = spearmanr(df.auc, df.popw_degrad_auc)
    print(f"WITHIN-CITY pooled: {r:+.3f} (p={p:.1e})")
    print(f"naive pooled:       {rp:+.3f} (p={pp:.1e})")


if __name__ == "__main__":
    main()

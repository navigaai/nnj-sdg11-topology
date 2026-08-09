"""Real flood-hazard disruption (open JRC global flood data).

Replaces the DEM low-elevation hazard proxy with a REAL 100-year river-flood
inundation scenario from the open JRC Global Flood Hazard maps (Dottori et al.;
floodMapGL_rp100y, ~1 km, water depth in m, EPSG:4326). For each city we sample the
flood depth at every network node, flag nodes in inundated cells (depth > threshold)
as flood-exposed, and run the existing progressive hazard-removal disruption on
those nodes. We report flood exposure per city and refit the morphology regression
under this real-flood scenario (reduced grid), for the cities with material exposure.

Writes output/flood_exposure.csv and output/regression_flood.csv.
Usage: uv run python scripts/flood_hazard.py
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import statsmodels.formula.api as smf
from pyproj import Transformer

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import access_points, load_greenspace, snap_points_to_nodes
from nnj_topology.data.network import load_walk_network
from nnj_topology.disruption.models import hazard_removal
from nnj_topology.districts.tiling import assign_nodes_to_hexes, district_resilience

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
FLOOD = Path("data/flood/floodMapGL_rp100y.tif")
MIN_DEPTH = 0.5     # m; cells with >=0.5 m water are treated as inundated
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_REP = 2
MORPH = ["intersection_density", "circuity", "orientation_entropy",
         "mean_street_length", "greenspace_fragmentation"]


def flood_nodes(graph, crs) -> set:
    """Nodes lying in cells with >= MIN_DEPTH m of 100-yr flood water."""
    with rasterio.open(FLOOD) as src:
        fcrs = src.crs
        band = src.read(1)
        nod = src.nodata
    nodes = list(graph.nodes)
    xs = np.array([float(graph.nodes[n]["x"]) for n in nodes])
    ys = np.array([float(graph.nodes[n]["y"]) for n in nodes])
    tr = Transformer.from_crs(crs, fcrs, always_xy=True)
    lon, lat = tr.transform(xs, ys)
    with rasterio.open(FLOOD) as src:
        rc = [src.index(x, y) for x, y in zip(lon, lat)]
    h, w = band.shape
    out = set()
    for n, (r, c) in zip(nodes, rc):
        if 0 <= r < h and 0 <= c < w:
            v = band[r, c]
            if (nod is None or v != nod) and np.isfinite(v) and v >= MIN_DEPTH:
                out.add(n)
    return out


def main() -> None:
    exp_rows, frames = [], []
    for city, (place, crs) in CITIES.items():
        g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
        g = clip_graph_to_boundary(g, load_urban_boundary(
            Path("data/ghsl/ghs_ucdb.gpkg"), city, crs), crs)
        g = add_travel_time(g)
        fn = flood_nodes(g, crs)
        frac = len(fn) / max(1, g.number_of_nodes())
        exp_rows.append({"city": city, "n_nodes": g.number_of_nodes(),
                         "flood_nodes": len(fn), "flood_frac": round(frac, 4)})
        print(f"  {city}: {len(fn)}/{g.number_of_nodes()} nodes flooded ({frac:.1%})")
        if len(fn) < 20:
            continue  # negligible exposure -> skip resilience run
        green = load_greenspace(place, crs, Path(f"data/{city}/green.gpkg"))
        src = snap_points_to_nodes(access_points(green), g)
        hn = assign_nodes_to_hexes(g, crs, 8)
        disrupt = functools.partial(hazard_removal, hazard_nodes=fn)
        res = district_resilience(
            g, lambda gg: accessibility_field(gg, src), disrupt,
            rhos=RHOS, n_replicates=N_REP, seed=42, hex_nodes=hn,
            max_dim=0, min_nodes=10, dim=0, min_persistence=1.0)
        frames.append(pd.DataFrame([{"city": city, "hex": c, "auc_flood": r.auc}
                                    for c, r in res.items()]))
    pd.DataFrame(exp_rows).to_csv("output/flood_exposure.csv", index=False)
    print("\n=== 100-yr flood exposure ===")
    print(pd.DataFrame(exp_rows).to_string(index=False))

    if not frames:
        print("\nNo city with material flood exposure; no regression.")
        return
    fl = pd.concat(frames, ignore_index=True)
    fl.to_csv("output/flood_auc.csv", index=False)
    # merge with the 8-city morphology table so materially-exposed new cities
    # (e.g. Vienna) are included, not just the original five.
    base = pd.read_csv("output/district_table_10city.csv")[["city", "hex"] + MORPH]
    df = fl.merge(base, on=["city", "hex"], how="inner")
    df.to_csv("output/regression_flood.csv", index=False)
    print(f"\n=== Flood-scenario regression (auc_flood ~ morphology + C(city)), "
          f"n={len(df)}, cities={df.city.nunique()} ===")
    m = smf.ols("auc_flood ~ " + " + ".join(MORPH + ["C(city)"]), data=df).fit()
    for t in MORPH:
        print(f"  {t:24s} coef={m.params[t]:+.4g}  p={m.pvalues[t]:.2g}")


if __name__ == "__main__":
    main()

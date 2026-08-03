"""Omitted-variable robustness (reviewer response).

Reviewers noted the district regression controls only for morphology and city
fixed effects, leaving several district attributes uncontrolled. This script adds
the controls that are computable from the data already in hand, WITHOUT recomputing
the expensive resilience metric (per-district AUC is reused from
output/district_table.csv):

  * green_area_km2   -- total green-space area in the district (green quantity/size)
  * major_road_share -- share of edges on the trunk/primary/secondary/tertiary
                        hierarchy (road hierarchy)
  * relief_m         -- std of node elevation within the district (topography)
  * centre_dist_km   -- distance from district centroid to the urban-centre
                        polygon centroid (city-centre proximity)

It then refits AUC ~ morphology + these controls + C(city) and reports whether the
headline morphology effects survive. Population/built density, land-use mix,
socioeconomic status and pedestrian-infrastructure quality require external data
sources and remain uncontrolled (stated in the manuscript Limitations).

Writes output/district_controls.csv and output/regression_controls.csv.
Usage: uv run python scripts/omitted_controls.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
import statsmodels.formula.api as smf
from pyproj import Transformer
from shapely.geometry import Polygon

from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import load_greenspace
from nnj_topology.data.network import load_walk_network
from nnj_topology.districts.tiling import assign_nodes_to_hexes

logging.getLogger().setLevel(logging.WARNING)


def _hex_polygon(cell: str, crs: str):
    """H3 cell boundary as a shapely polygon reprojected to the metric ``crs``."""
    ring = h3.cell_to_boundary(cell)
    poly = Polygon([(lng, lat) for lat, lng in ring])
    return gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(crs).iloc[0]

CITIES = {
    "istanbul": ("İstanbul, Turkey", "EPSG:32635"),
    "barcelona": ("Barcelona, Spain", "EPSG:25831"),
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
    "bogota": ("Bogotá, Colombia", "EPSG:32618"),
    "phoenix": ("Phoenix, Arizona, USA", "EPSG:26912"),
}
MAJOR = {"trunk", "primary", "secondary", "tertiary",
         "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
MORPH = ["intersection_density", "circuity", "orientation_entropy",
         "mean_street_length", "greenspace_fragmentation"]
CONTROLS = ["green_area_km2", "major_road_share", "relief_m", "centre_dist_km"]
OUT = Path("output")


def _is_major(hw) -> bool:
    if hw is None:
        return False
    if isinstance(hw, list):
        return any(h in MAJOR for h in hw)
    return str(hw).strip("[]'\" ").split(",")[0] in MAJOR


def _node_elevations(graph, crs: str, dem_path: Path) -> dict:
    """Sample DEM elevation at each node; returns {node: elevation}."""
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        band = src.read(1).astype("float64")
        nodata = src.nodata
    nodes = list(graph.nodes)
    xs = np.array([float(graph.nodes[n]["x"]) for n in nodes])
    ys = np.array([float(graph.nodes[n]["y"]) for n in nodes])
    tr = Transformer.from_crs(crs, dem_crs, always_xy=True)
    lon, lat = tr.transform(xs, ys)
    with rasterio.open(dem_path) as src:
        rc = [src.index(x, y) for x, y in zip(lon, lat)]
    elev = {}
    h, w = band.shape
    for n, (r, c) in zip(nodes, rc):
        if 0 <= r < h and 0 <= c < w:
            v = band[r, c]
            if nodata is None or v != nodata:
                elev[n] = float(v)
    return elev


def city_controls(city: str, place: str, crs: str) -> pd.DataFrame:
    graph = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
    boundary = load_urban_boundary(Path("data/ghsl/ghs_ucdb.gpkg"), city, crs)
    graph = clip_graph_to_boundary(graph, boundary, crs)
    green = load_greenspace(place, crs, Path(f"data/{city}/green.gpkg"))
    hex_nodes = assign_nodes_to_hexes(graph, crs, 8)
    elev = _node_elevations(graph, crs, Path(f"data/{city}/dem.tif"))
    centre = boundary.centroid

    rows = []
    for cell, nodes in hex_nodes.items():
        sub = graph.subgraph(nodes)
        if sub.number_of_edges() == 0:
            continue
        hexpoly = _hex_polygon(cell, crs)
        clipped = green[green.intersects(hexpoly)].geometry.intersection(hexpoly)
        green_area = float(clipped.area.sum()) / 1e6 if len(clipped) else 0.0
        major = sum(_is_major(d.get("highway")) for _, _, d in sub.edges(data=True))
        share = major / sub.number_of_edges()
        evs = [elev[n] for n in nodes if n in elev]
        relief = float(np.std(evs)) if len(evs) >= 3 else 0.0
        cdist = float(hexpoly.centroid.distance(centre)) / 1000.0
        rows.append({"city": city, "hex": cell, "green_area_km2": green_area,
                     "major_road_share": share, "relief_m": relief,
                     "centre_dist_km": cdist})
    print(f"  {city}: {len(rows)} districts with controls")
    return pd.DataFrame(rows)


def main() -> None:
    controls = pd.concat(
        [city_controls(c, p, crs) for c, (p, crs) in CITIES.items()],
        ignore_index=True,
    )
    controls.to_csv(OUT / "district_controls.csv", index=False)

    base = pd.read_csv(OUT / "district_table.csv")
    df = base.merge(controls, on=["city", "hex"], how="inner")
    print(f"\nmerged n = {len(df)} (base {len(base)})")

    rhs_base = " + ".join(MORPH + ["C(city)"])
    rhs_full = " + ".join(MORPH + CONTROLS + ["C(city)"])
    m0 = smf.ols(f"auc ~ {rhs_base}", data=df).fit()
    m1 = smf.ols(f"auc ~ {rhs_full}", data=df).fit()

    rows = []
    for term in MORPH + CONTROLS:
        rows.append({
            "term": term,
            "coef_base": float(m0.params[term]) if term in m0.params else np.nan,
            "p_base": float(m0.pvalues[term]) if term in m0.pvalues else np.nan,
            "coef_full": float(m1.params[term]),
            "p_full": float(m1.pvalues[term]),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "regression_controls.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n=== AUC ~ morphology (+controls) + C(city) ===")
        print(res.to_string(index=False))
    print(f"\nR2: base {m0.rsquared:.4f} -> +controls {m1.rsquared:.4f}")


if __name__ == "__main__":
    main()

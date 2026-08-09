"""Green-space definition sensitivity (reviewer B).

The headline uses an inclusive green-space set (park, garden, recreation_ground,
grass, square, wood). Reviewers note that some of these are not reliably public or
accessible (private gardens, fenced grass, impenetrable woods). We re-download the
green spaces WITH their OSM tags and recompute the resilience metric under a STRICT,
accessibility-conservative subset -- keep only leisure=park, leisure/landuse=
recreation_ground and place=square; drop natural=wood, landuse=grass and
leisure=garden -- then compare the district regression against the inclusive set on
the same reduced disruption grid.

Writes output/green_sensitivity.csv. Usage: uv run python scripts/green_sensitivity.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import (
    GREENSPACE_TAGS,
    access_points,
    snap_points_to_nodes,
)
from nnj_topology.data.network import load_walk_network
from nnj_topology.districts.tiling import assign_nodes_to_hexes, district_resilience
from nnj_topology.disruption.models import random_removal

logging.getLogger().setLevel(logging.WARNING)

CITIES = {
    "istanbul": ("İstanbul, Turkey", "EPSG:32635"),
    "barcelona": ("Barcelona, Spain", "EPSG:25831"),
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
    "bogota": ("Bogotá, Colombia", "EPSG:32618"),
    "phoenix": ("Phoenix, Arizona, USA", "EPSG:26912"),
}
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_REP = 2
MORPH = ["intersection_density", "circuity", "orientation_entropy",
         "mean_street_length", "greenspace_fragmentation"]


def _tagged_green(city: str, place: str, crs: str) -> gpd.GeoDataFrame:
    """Green polygons WITH tag columns (cached to green_tagged.gpkg)."""
    cache = Path(f"data/{city}/green_tagged.gpkg")
    if cache.exists():
        return gpd.read_file(cache).to_crs(crs)
    gdf = ox.features_from_place(place, tags=GREENSPACE_TAGS)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(crs)
    keep = [c for c in ["leisure", "landuse", "natural", "place", "geometry"]
            if c in gdf.columns]
    gdf = gdf.reset_index(drop=True)[keep]
    gdf.to_file(cache, driver="GPKG")
    return gdf


def _strict(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Accessibility-conservative subset: keep park / recreation_ground / square;
    drop wood, grass, garden."""
    lei = gdf.get("leisure", pd.Series(index=gdf.index, dtype=object)).fillna("")
    lu = gdf.get("landuse", pd.Series(index=gdf.index, dtype=object)).fillna("")
    pl = gdf.get("place", pd.Series(index=gdf.index, dtype=object)).fillna("")
    keep = (lei.isin(["park", "recreation_ground"]) |
            (lu == "recreation_ground") | (pl == "square"))
    return gdf[keep].reset_index(drop=True)


def _auc_frame(g, green, crs, hn, label) -> pd.DataFrame:
    src = snap_points_to_nodes(access_points(green), g)
    res = district_resilience(
        g, lambda gg: accessibility_field(gg, src), random_removal,
        rhos=RHOS, n_replicates=N_REP, seed=42, hex_nodes=hn,
        max_dim=0, min_nodes=10, dim=0, min_persistence=1.0)
    return pd.DataFrame([{"hex": c, f"auc_{label}": r.auc} for c, r in res.items()])


def main() -> None:
    frames = []
    for city, (place, crs) in CITIES.items():
        g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
        g = clip_graph_to_boundary(g, load_urban_boundary(
            Path("data/ghsl/ghs_ucdb.gpkg"), city, crs), crs)
        g = add_travel_time(g)
        gt = _tagged_green(city, place, crs)
        hn = assign_nodes_to_hexes(g, crs, 8)
        allg = _auc_frame(g, gt[["geometry"]], crs, hn, "all")
        strg = _auc_frame(g, _strict(gt), crs, hn, "strict")
        m = allg.merge(strg, on="hex")
        m["city"] = city
        print(f"  {city}: {len(gt)} green ({len(_strict(gt))} strict), {len(m)} districts")
        frames.append(m)
    ga = pd.concat(frames, ignore_index=True)
    ga.to_csv("output/green_sensitivity.csv", index=False)

    df = pd.read_csv("output/district_table.csv").merge(ga, on=["city", "hex"])
    print(f"\nmerged n={len(df)}")
    print("Spearman(auc_all, auc_strict) =",
          round(spearmanr(df.auc_all, df.auc_strict).correlation, 3))
    for tgt in ["auc_all", "auc_strict"]:
        m = smf.ols(f"{tgt} ~ " + " + ".join(MORPH + ["C(city)"]), data=df).fit()
        print(f"\n=== {tgt} ~ morphology + C(city) ===")
        for t in MORPH:
            print(f"  {t:24s} coef={m.params[t]:+.4g}  p={m.pvalues[t]:.2g}")


if __name__ == "__main__":
    main()

"""Green-space class-exclusion sensitivity (reviewer): drop wood / grass / garden
one at a time and refit, to see whether any single questionable class drives the
result. Five-city core (cities with cached tagged green), reduced grid.

Writes output/green_individual.csv. Usage: uv run python scripts/green_individual.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import statsmodels.formula.api as smf

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.greenspace import access_points, snap_points_to_nodes
from nnj_topology.data.network import load_walk_network
from nnj_topology.disruption.models import random_removal
from nnj_topology.districts.tiling import assign_nodes_to_hexes, district_resilience

logging.getLogger().setLevel(logging.WARNING)
CITIES = {
    "istanbul": ("İstanbul, Turkey", "EPSG:32635"),
    "barcelona": ("Barcelona, Spain", "EPSG:25831"),
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
    "bogota": ("Bogotá, Colombia", "EPSG:32618"),
    "phoenix": ("Phoenix, Arizona, USA", "EPSG:26912"),
}
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
MORPH = ["intersection_density", "circuity", "orientation_entropy",
         "mean_street_length", "greenspace_fragmentation"]
HEAD = ["circuity", "mean_street_length", "orientation_entropy"]


def _drop(gt, col, val):
    if col not in gt.columns:
        return gt
    keep = gt[col].fillna("") != val
    return gt[keep].reset_index(drop=True)


def _auc(g, green, crs, hn, label):
    src = snap_points_to_nodes(access_points(green), g)
    res = district_resilience(g, lambda gg: accessibility_field(gg, src), random_removal,
                              rhos=RHOS, n_replicates=2, seed=42, hex_nodes=hn,
                              max_dim=0, min_nodes=10, dim=0, min_persistence=1.0)
    return pd.DataFrame([{"hex": c, f"auc_{label}": r.auc} for c, r in res.items()])


def main() -> None:
    frames = []
    for city, (place, crs) in CITIES.items():
        g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
        g = clip_graph_to_boundary(g, load_urban_boundary(
            Path("data/ghsl/ghs_ucdb.gpkg"), city, crs), crs)
        g = add_travel_time(g)
        gt = gpd.read_file(Path(f"data/{city}/green_tagged.gpkg")).to_crs(crs)
        hn = assign_nodes_to_hexes(g, crs, 8)
        m = _auc(g, gt[["geometry"]], crs, hn, "all")
        for lab, col, val in [("no_wood", "natural", "wood"),
                              ("no_grass", "landuse", "grass"),
                              ("no_garden", "leisure", "garden")]:
            m = m.merge(_auc(g, _drop(gt, col, val), crs, hn, lab), on="hex")
        m["city"] = city
        frames.append(m)
        print(f"  {city} done")
    ga = pd.concat(frames, ignore_index=True)
    base = pd.read_csv("output/district_table.csv")[["city", "hex"] + MORPH]
    df = ga.merge(base, on=["city", "hex"])
    df.to_csv("output/green_individual.csv", index=False)
    print(f"\nn={len(df)}. Headline coefficients under each exclusion:")
    print(f"{'variant':10s} " + "  ".join(f"{h[:12]:>12s}" for h in HEAD))
    for lab in ["all", "no_wood", "no_grass", "no_garden"]:
        mm = smf.ols(f"auc_{lab} ~ " + " + ".join(MORPH + ["C(city)"]), data=df).fit()
        print(f"{lab:10s} " + "  ".join(f"{mm.params[h]:>+12.4g}" for h in HEAD)
              + "   p: " + ", ".join(f"{mm.pvalues[h]:.1e}" for h in HEAD))


if __name__ == "__main__":
    main()

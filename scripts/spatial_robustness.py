"""Spatial-dependence robustness checks for the district-level regression.

Addresses two reviewer concerns:
  (1) the 2,603 districts are not independent observations (they nest within
      5 cities and are spatially contiguous), so classical OLS standard errors
      are too small;
  (2) spatial autocorrelation of the residuals was never quantified.

Outputs:
  - output/regression_clustered.csv : coefficients with classical, HC3 and
    city-clustered standard errors side by side;
  - output/moran_residuals.csv      : global Moran's I on the pooled residuals
    and within each city, using H3 first-ring adjacency as the spatial weights.

Usage: uv run python scripts/spatial_robustness.py
"""
from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import pandas as pd
from libpysal.weights import W
from esda.moran import Moran

from nnj_topology.analysis.regression import (
    city_clustered_regression,
    fixed_effects_regression,
)

FEATURES = [
    "intersection_density",
    "circuity",
    "orientation_entropy",
    "mean_street_length",
    "greenspace_fragmentation",
]
OUT = Path("output")


def clustered_table(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficients with three standard-error flavours for comparison."""
    classical = fixed_effects_regression(df, "auc", FEATURES)
    hc3 = classical.get_robustcov_results(cov_type="HC3")
    clustered = city_clustered_regression(df, "auc", FEATURES)

    rows = []
    for term in classical.params.index:
        if term.startswith("C(city)") or term == "Intercept":
            continue
        rows.append(
            {
                "term": term,
                "coef": float(classical.params[term]),
                "se_classical": float(classical.bse[term]),
                "p_classical": float(classical.pvalues[term]),
                "se_hc3": float(hc3.bse[list(classical.params.index).index(term)]),
                "se_cluster_city": float(clustered.bse[term]),
                "p_cluster_city": float(clustered.pvalues[term]),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "regression_clustered.csv", index=False)
    return tab


def _hex_weights(hexes: list[str]) -> W:
    """First-ring H3 adjacency as libpysal spatial weights (queen-like)."""
    hexset = set(hexes)
    neighbors = {}
    for hx in hexes:
        ring = [n for n in h3.grid_disk(hx, 1) if n != hx and n in hexset]
        neighbors[hx] = ring
    return W(neighbors, silence_warnings=True)


def moran_table(df: pd.DataFrame) -> pd.DataFrame:
    """Global Moran's I on FE-regression residuals: pooled and per city."""
    result = fixed_effects_regression(df, "auc", FEATURES)
    df = df.copy()
    df["resid"] = result.resid.values

    rows = []
    # Per city (spatial contiguity is only meaningful within a city).
    for city, g in df.groupby("city"):
        g = g.reset_index(drop=True)
        w = _hex_weights(list(g["hex"]))
        # keep only islands-free component
        w.transform = "r"
        mask = np.array([len(w.neighbors[h]) > 0 for h in g["hex"]])
        if mask.sum() < 20:
            continue
        gm = g[mask].reset_index(drop=True)
        wm = _hex_weights(list(gm["hex"]))
        wm.transform = "r"
        mi = Moran(gm["resid"].values, wm, permutations=999)
        rows.append(
            {
                "scope": city,
                "n": int(mask.sum()),
                "morans_I": float(mi.I),
                "expected_I": float(mi.EI),
                "p_sim": float(mi.p_sim),
                "z_sim": float(mi.z_sim),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "moran_residuals.csv", index=False)
    return tab


def main() -> None:
    df = pd.read_csv(OUT / "district_table.csv")
    print("=== Coefficients with clustered SE (target=auc, res 8) ===")
    ct = clustered_table(df)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(ct.to_string(index=False))
    print("\n=== Moran's I on residuals (H3 first-ring adjacency) ===")
    mt = moran_table(df)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(mt.to_string(index=False))


if __name__ == "__main__":
    main()

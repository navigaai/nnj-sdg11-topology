"""Estimated spatial regressions (reviewer response).

The Moran's I diagnostic (scripts/spatial_robustness.py) showed the fixed-effects
OLS residuals are spatially autocorrelated within every city. This script goes
one step further and *estimates* spatial models that absorb that dependence,
checking whether the morphology conclusion survives:

  * OLS with city fixed effects (baseline, with spatial diagnostics);
  * Spatial Error model  (SEM): y = Xb + u, u = lambda W u + e;
  * Spatial Lag model    (SAR): y = rho W y + Xb + e.

Weights W are first-ring H3 adjacency built BLOCK-DIAGONAL by city (contiguity is
only meaningful within a city; there are no cross-city links), row-standardised.
Islands (districts with no within-city neighbour) are dropped consistently from
y, X and W. City fixed effects enter as dummy columns (Amsterdam = reference).

Writes output/spatial_regression.csv and prints a summary.
Usage: uv run python scripts/spatial_regression.py
"""
from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import pandas as pd
from libpysal.weights import W
from spreg import OLS, ML_Error, ML_Lag

FEATURES = [
    "intersection_density",
    "circuity",
    "orientation_entropy",
    "mean_street_length",
    "greenspace_fragmentation",
]
REF_CITY = "amsterdam"  # dropped dummy (reference level)
OUT = Path("output")


def build_block_weights(df: pd.DataFrame) -> tuple[W, np.ndarray]:
    """First-ring H3 adjacency, block-diagonal by city; returns (W, keep_mask).

    keep_mask drops islands (no within-city neighbour) so the estimator gets a
    connected-enough graph. Indices in the returned W are 0..m-1 over kept rows.
    """
    hexes = df["hex"].to_numpy()
    cities = df["city"].to_numpy()
    hex_to_idx = {h: i for i, h in enumerate(hexes)}
    raw_neighbors: dict[int, list[int]] = {}
    for i, (h, c) in enumerate(zip(hexes, cities)):
        ring = []
        for nb in h3.grid_disk(h, 1):
            if nb == h:
                continue
            j = hex_to_idx.get(nb)
            if j is not None and cities[j] == c:  # same-city links only
                ring.append(j)
        raw_neighbors[i] = ring
    keep = np.array([len(raw_neighbors[i]) > 0 for i in range(len(df))])
    old_to_new = {old: new for new, old in enumerate(np.where(keep)[0])}
    neighbors = {
        old_to_new[i]: [old_to_new[j] for j in raw_neighbors[i] if j in old_to_new]
        for i in range(len(df))
        if keep[i]
    }
    w = W(neighbors, silence_warnings=True)
    w.transform = "r"
    return w, keep


def _design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Morphology features + city dummies (Amsterdam reference)."""
    X = [df[FEATURES].to_numpy()]
    names = list(FEATURES)
    for c in sorted(df["city"].unique()):
        if c == REF_CITY:
            continue
        X.append((df["city"] == c).to_numpy().astype(float).reshape(-1, 1))
        names.append(f"city_{c}")
    return np.hstack(X), names


def main() -> None:
    df = pd.read_csv(OUT / "district_table.csv")
    w, keep = build_block_weights(df)
    d = df[keep].reset_index(drop=True)
    y = d["auc"].to_numpy().reshape(-1, 1)
    X, names = _design(d)
    print(f"n kept = {len(d)} (dropped {int((~keep).sum())} islands); "
          f"weights: mean neighbours = {np.mean([len(v) for v in w.neighbors.values()]):.2f}")

    ols = OLS(y, X, w=w, name_x=names, name_y="auc", spat_diag=True, moran=True)
    sem = ML_Error(y, X, w=w, name_x=names, name_y="auc")
    sar = ML_Lag(y, X, w=w, name_x=names, name_y="auc")

    rows = []
    for label, m in [("OLS_FE", ols), ("SpatialError", sem), ("SpatialLag", sar)]:
        b = np.asarray(m.betas).flatten()
        names_full = ["CONSTANT"] + names
        # ML models append the spatial parameter (lambda/rho) at the end of betas
        spatial_val = None
        if label in ("SpatialError", "SpatialLag"):
            spatial_val = float(b[-1])
            b_core = b[:-1]
        else:
            b_core = b
        se = np.sqrt(np.diag(m.vm)).flatten()
        z = b_core / se[: len(b_core)]
        for nm, coef, zz in zip(names_full, b_core, z):
            if nm in FEATURES or nm == "CONSTANT":
                rows.append({"model": label, "term": nm, "coef": float(coef),
                             "z": float(zz)})
        if spatial_val is not None:
            rows.append({"model": label,
                         "term": ("lambda" if label == "SpatialError" else "rho"),
                         "coef": spatial_val, "z": np.nan})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "spatial_regression.csv", index=False)

    # Pretty print morphology coefficients across the three models
    print("\n=== Morphology coefficients (z-stat) across models ===")
    piv = res[res["term"].isin(FEATURES)].pivot(index="term", columns="model",
                                                values="coef")
    zpiv = res[res["term"].isin(FEATURES)].pivot(index="term", columns="model",
                                                 values="z")
    for t in FEATURES:
        parts = [f"{t:26s}"]
        for mdl in ["OLS_FE", "SpatialError", "SpatialLag"]:
            parts.append(f"{mdl}: {piv.loc[t, mdl]:+.4g} (z={zpiv.loc[t, mdl]:+.1f})")
        print("  ".join(parts))
    print("\nSpatial parameters:")
    print(res[res["term"].isin(["lambda", "rho"])].to_string(index=False))
    mr = getattr(ols, "moran_res", None)
    if mr is not None:
        print(f"\nMoran's I on OLS residuals (spreg): I={mr[0]:.4f}, z={mr[1]:.3f}, p={mr[2]:.4g}")


if __name__ == "__main__":
    main()

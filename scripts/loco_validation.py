"""Leave-one-city-out (LOCO) out-of-sample validation of cross-city transfer
(reviewer: only 8 independent clusters; are the coefficients generalisable beyond the
cities used to fit them?).

The headline model identifies morphology from WITHIN-city variation (city fixed
effects). The corresponding out-of-sample question is: do the within-city slopes learned
from seven cities predict the ordering of districts in a held-out eighth city it never
saw? For each held-out city:
  1. city-demean (within-transform) predictors and AUC on the 7 training cities,
  2. fit AUC_within ~ morphology_within (no intercept: FE already removed),
  3. apply those slopes to the held-out city's OWN city-demeaned predictors,
  4. Spearman(predicted within-city AUC, actual within-city AUC) in the held-out city.

A positive, consistent out-of-sample Spearman is genuine cross-city transfer; the FE
intercept for the held-out city is not needed because the target is the within-city
ranking. Reports per-held-out-city and the mean.

Reads output/district_table.csv. Writes output/loco_validation.csv.
Usage: uv run python scripts/loco_validation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HEAD = ["circuity", "mean_street_length", "orientation_entropy"]
ALL = HEAD + ["intersection_density", "greenspace_fragmentation"]


def _within(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols + ["auc"]:
        out[c + "_w"] = df[c] - df.groupby("city")[c].transform("mean")
    return out


def main() -> None:
    df = pd.read_csv("output/district_table.csv")
    cities = sorted(df.city.unique())
    rows = []
    for held in cities:
        train = df[df.city != held]
        test = df[df.city == held].copy()
        tw = _within(train, ALL)
        X = tw[[c + "_w" for c in ALL]].to_numpy()
        y = tw["auc_w"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)   # within-city slopes, 7 cities
        # apply to held-out city's OWN within-transform
        for c in ALL:
            test[c + "_w"] = test[c] - test[c].mean()
        test["auc_w"] = test["auc"] - test["auc"].mean()
        pred = test[[c + "_w" for c in ALL]].to_numpy() @ beta
        r, p = spearmanr(pred, test["auc_w"].to_numpy())
        rows.append({"held_out": held, "n": len(test),
                     "oos_spearman": round(float(r), 3), "p": float(p)})
        print(f"  held-out {held:12s} n={len(test):4d}  out-of-sample rho={r:+.3f} (p={p:.2g})")
    res = pd.DataFrame(rows)
    res.to_csv("output/loco_validation.csv", index=False)
    m = res.oos_spearman.mean()
    pos = int((res.oos_spearman > 0).sum())
    print(f"\nmean out-of-sample Spearman = {m:+.3f}; positive in {pos}/{len(res)} held-out cities")
    print("wrote output/loco_validation.csv")


if __name__ == "__main__":
    main()

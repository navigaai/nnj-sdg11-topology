"""Extend the distance/connectivity benchmark validation from the five-city core
to the full eight-city headline sample (reviewer: validation sample must match the
analysis sample). Computes ONLY the three added cities (Singapore, Nairobi, Vienna),
appends them to output/benchmark_validation.csv, and recomputes the eight-city
convergent- and within-city criterion-validity statistics.

Reuses scripts/benchmark_validation.city_baselines (no persistent homology or
Wasserstein: reduced grid, cheap). Usage: uv run python scripts/benchmark_extend.py
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from benchmark_validation import city_baselines

NEW = {
    "singapore": ("Singapore", "EPSG:32648"),
    "nairobi": ("Nairobi, Kenya", "EPSG:32737"),
    "vienna": ("Vienna, Austria", "EPSG:32633"),
}


def main() -> None:
    base = pd.read_csv("output/benchmark_validation.csv")
    have = set(base.city.unique())
    todo = {c: v for c, v in NEW.items() if c not in have}
    if todo:
        add = pd.concat([city_baselines(c, p, crs) for c, (p, crs) in todo.items()],
                        ignore_index=True)
        base = pd.concat([base, add], ignore_index=True)
        base.to_csv("output/benchmark_validation.csv", index=False)
    print(f"benchmark cities now: {sorted(base.city.unique())} (n={len(base)})")

    d = pd.read_csv("output/district_table.csv")
    df = d.merge(base, on=["city", "hex"], how="inner")
    print(f"merged eight-city n = {len(df)}")

    # convergent validity (naive pooled) on 8 cities
    print("\n=== Eight-city convergent validity: Spearman(topological AUC, baseline) ===")
    for col, label in [("degrad_access_auc", "M1 distance-degradation"),
                       ("degrad_unreach_auc", "M2 connectivity-degradation")]:
        r, p = spearmanr(df["auc"], df[col])
        print(f"  {label:32s}: rho={r:+.3f} (p={p:.2g})")

    # within-city criterion validity (the level the FE regression identifies at)
    print("\n=== Eight-city WITHIN-city criterion validity (per city + pooled) ===")
    rows = []
    for c, g in df.groupby("city"):
        r, p = spearmanr(g["auc"], g["degrad_access_auc"])
        rows.append({"scope": c, "n": len(g), "spearman_distance": round(float(r), 3),
                     "p": float(p)})
        print(f"  {c:12s} n={len(g):4d}  rho={r:+.3f} (p={p:.2g})")
    for col in ["auc", "degrad_access_auc"]:
        df[col + "_wr"] = df.groupby("city")[col].rank()
    r, p = spearmanr(df["auc_wr"], df["degrad_access_auc_wr"])
    rows.append({"scope": "WITHIN-CITY pooled", "n": len(df),
                 "spearman_distance": round(float(r), 3), "p": float(p)})
    print(f"  {'WITHIN pooled':12s} n={len(df):4d}  rho={r:+.3f} (p={p:.2g})")
    rn, pn = spearmanr(df["auc"], df["degrad_access_auc"])
    rows.append({"scope": "NAIVE pooled", "n": len(df),
                 "spearman_distance": round(float(rn), 3), "p": float(pn)})
    pd.DataFrame(rows).to_csv("output/criterion_validity_8city.csv", index=False)
    print("\nwrote output/criterion_validity_8city.csv")


if __name__ == "__main__":
    main()

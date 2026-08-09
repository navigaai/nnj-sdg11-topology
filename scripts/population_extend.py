"""Extend the population-weighted criterion validation from the five-city core to the
full eight-city headline sample (reviewer: validation sample must match the analysis
sample). Computes ONLY the three added cities (Singapore, Nairobi, Vienna) with the
global GHS-POP R2023A grid, appends to output/population_validation.csv, and recomputes
the eight-city within-city criterion validity.

Reuses scripts/population_validation.city_pop_degrad.
Usage: uv run python scripts/population_extend.py
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from population_validation import city_pop_degrad

NEW = {
    "singapore": ("Singapore", "EPSG:32648"),
    "nairobi": ("Nairobi, Kenya", "EPSG:32737"),
    "vienna": ("Vienna, Austria", "EPSG:32633"),
}


def main() -> None:
    pv = pd.read_csv("output/population_validation.csv")
    have = set(pv.city.unique())
    todo = {c: v for c, v in NEW.items() if c not in have}
    if todo:
        add = pd.concat([city_pop_degrad(c, p, crs) for c, (p, crs) in todo.items()],
                        ignore_index=True)
        pv = pd.concat([pv, add], ignore_index=True)
        pv.to_csv("output/population_validation.csv", index=False)
    print(f"population cities now: {sorted(pv.city.unique())} (n={len(pv)})")

    df = pd.read_csv("output/district_table.csv").merge(pv, on=["city", "hex"])
    print(f"merged eight-city n = {len(df)}")

    print("\n=== Eight-city criterion validity vs POPULATION-WEIGHTED degradation ===")
    rows = []
    for c, gg in df.groupby("city"):
        r, p = spearmanr(gg.auc, gg.popw_degrad_auc)
        rows.append({"scope": c, "n": len(gg), "spearman_popw": round(float(r), 3),
                     "p": float(p)})
        print(f"  {c:12s} n={len(gg):4d}  rho={r:+.3f} (p={p:.2g})")
    for col in ["auc", "popw_degrad_auc"]:
        df[col + "_wr"] = df.groupby("city")[col].rank()
    r, p = spearmanr(df.auc_wr, df.popw_degrad_auc_wr)
    rows.append({"scope": "WITHIN-CITY pooled", "n": len(df),
                 "spearman_popw": round(float(r), 3), "p": float(p)})
    print(f"  {'WITHIN pooled':12s} n={len(df):4d}  rho={r:+.3f} (p={p:.2g})")
    rn, pn = spearmanr(df.auc, df.popw_degrad_auc)
    rows.append({"scope": "NAIVE pooled", "n": len(df),
                 "spearman_popw": round(float(rn), 3), "p": float(pn)})
    pd.DataFrame(rows).to_csv("output/population_validity_8city.csv", index=False)
    print("\nwrote output/population_validity_8city.csv")


if __name__ == "__main__":
    main()

"""Criterion (convergent) validity of the topological resilience metric, at the
level at which it is actually used -- WITHIN city (reviewer W1).

The headline regression identifies the morphology effects from within-city variation
(city fixed effects). The relevant convergent-validity question is therefore also
within-city: does the topological district AUC track a simple distance-based
accessibility-degradation benchmark (scripts/benchmark_validation.py) once each
city's overall AUC level is removed? The naive pooled correlation is depressed by
cross-city AUC scale differences -- exactly the scale the city FE absorbs.

Writes output/criterion_validity.csv. Usage: uv run python scripts/criterion_validity.py
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    b = pd.read_csv("output/benchmark_validation.csv")
    d = pd.read_csv("output/district_table.csv")[["city", "hex", "auc"]]
    df = d.merge(b, on=["city", "hex"], how="inner")

    rows = []
    for c, g in df.groupby("city"):
        r, p = spearmanr(g["auc"], g["degrad_access_auc"])
        rows.append({"scope": c, "n": len(g), "spearman_distance": round(float(r), 3),
                     "p": float(p)})
    # within-city pooled (rank within city, then correlate)
    for col in ["auc", "degrad_access_auc"]:
        df[col + "_wr"] = df.groupby("city")[col].rank()
    r, p = spearmanr(df["auc_wr"], df["degrad_access_auc_wr"])
    rp, pp = spearmanr(df["auc"], df["degrad_access_auc"])
    rows.append({"scope": "WITHIN-CITY pooled", "n": len(df),
                 "spearman_distance": round(float(r), 3), "p": float(p)})
    rows.append({"scope": "NAIVE pooled", "n": len(df),
                 "spearman_distance": round(float(rp), 3), "p": float(pp)})

    res = pd.DataFrame(rows)
    res.to_csv("output/criterion_validity.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

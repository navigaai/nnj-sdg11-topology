"""Cross-city slope heterogeneity and the pooled-model assumption (reviewer: city fixed
effects control mean differences but do NOT show the morphology->resilience slope is the
same across cities; the pooled coefficient should be read as an average conditional
association, not a universal effect).

Three things are reported:
  1. A formal test that slopes differ across cities: OLS with morphology x city
     interactions vs the fixed-effects-only model, joint F-test on the headline
     interaction terms (H0: all headline slopes equal across cities).
  2. The empirical distribution of per-city slopes (mean, SD, coefficient of variation,
     min/max, sign agreement) for each headline descriptor.
  3. A random-slopes mixed model (AUC ~ headline + (headline | city)): the fixed (mean)
     slope alongside the estimated between-city SD of the slope, so the pooled
     coefficient is explicitly the centre of a distribution of city-specific slopes.

Reads output/district_table.csv. Writes output/slope_heterogeneity.csv.
Usage: uv run python scripts/slope_heterogeneity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HEAD = ["circuity", "mean_street_length", "orientation_entropy"]
ALL = HEAD + ["intersection_density", "greenspace_fragmentation"]


def main() -> None:
    df = pd.read_csv("output/district_table.csv")

    # 1) joint test that headline slopes differ across cities
    m_fe = smf.ols("auc ~ " + " + ".join(ALL + ["C(city)"]), data=df).fit()
    m_ix = smf.ols("auc ~ " + " + ".join(ALL) + " + C(city) + "
                   + " + ".join(f"C(city):{t}" for t in HEAD), data=df).fit()
    ix_terms = [c for c in m_ix.params.index
                if any(c.endswith(":" + t) or c.startswith(t + ":") for t in HEAD)
                and "C(city)" in c]
    ftest = m_ix.f_test(" = 0, ".join(ix_terms) + " = 0")
    print("=== (1) Do headline slopes differ across cities? ===")
    print(f"joint F-test on {len(ix_terms)} morphology x city interactions: "
          f"F={float(ftest.fvalue):.2f}, p={float(ftest.pvalue):.2g}")
    print("  -> slopes are NOT identical across cities; pooled slope = average "
          "conditional association" if float(ftest.pvalue) < 0.05
          else "  -> no significant slope heterogeneity")

    # 2) empirical per-city slope distribution
    print("\n=== (2) Per-city slope distribution (separate OLS per city) ===")
    per = {t: [] for t in HEAD}
    for _, g in df.groupby("city"):
        mm = smf.ols("auc ~ " + " + ".join(ALL), data=g).fit()
        for t in HEAD:
            per[t].append(float(mm.params[t]))
    rows = []
    for t in HEAD:
        a = np.array(per[t])
        pooled = float(m_fe.params[t])
        cv = float(np.std(a) / abs(np.mean(a))) if np.mean(a) != 0 else np.nan
        agree = int((np.sign(a) == np.sign(pooled)).sum())
        print(f"  {t:20s} pooled={pooled:+.4g}  city mean={a.mean():+.4g} "
              f"SD={a.std():.4g}  CV={cv:.2f}  range=[{a.min():+.3g},{a.max():+.3g}]  "
              f"sign agree {agree}/8")
        rows.append({"term": t, "pooled": round(pooled, 4),
                     "city_mean_slope": round(float(a.mean()), 4),
                     "city_slope_sd": round(float(a.std()), 4),
                     "cv": round(cv, 2), "sign_agree_of_8": agree})

    # 3) random-slopes mixed model
    print("\n=== (3) Random-slopes mixed model: fixed slope +/- between-city SD ===")
    try:
        z = df.copy()
        for t in HEAD:
            z[t] = (z[t] - z[t].mean()) / z[t].std()
        z["auc_s"] = (z["auc"] - z["auc"].mean()) / z["auc"].std()
        md = smf.mixedlm("auc_s ~ " + " + ".join(HEAD), z, groups=z["city"],
                         re_formula="~" + " + ".join(HEAD))
        mf = md.fit(method="lbfgs", maxiter=200)
        cov = mf.cov_re
        for t in HEAD:
            fx = float(mf.fe_params[t])
            sd = float(np.sqrt(cov.loc[t, t])) if t in cov.index else np.nan
            print(f"  {t:20s} fixed(mean) slope={fx:+.3f} (std units)  "
                  f"between-city SD={sd:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  mixed model did not converge cleanly: {exc}")

    pd.DataFrame(rows).to_csv("output/slope_heterogeneity.csv", index=False)
    print("\nwrote output/slope_heterogeneity.csv")


if __name__ == "__main__":
    main()

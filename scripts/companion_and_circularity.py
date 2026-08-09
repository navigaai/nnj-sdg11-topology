"""Two reviewer points, from already-computed outputs (no new simulation):

#6 Disconnection channel as a COMPANION outcome. The H0/Wasserstein AUC measures graded
   persistence of the CONNECTED served structure; nodes that lose all access leave the
   finite diagram. So a connectivity-loss outcome (unreachable-fraction degradation,
   degrad_unreach_auc, already computed for all 8 cities) is reported as a SEPARATE,
   complementary outcome: what morphology predicts about outright disconnection, and how
   weakly it relates to the topological AUC.

#4 Shared-origin / circularity. Morphology descriptors and the topological AUC are both
   derived from the same network, so part of the association could be two summaries of
   the same object (circuity <-> H0 persistence especially). Partial check: does the
   morphology signal survive controlling for a same-network measure of the district's
   raw topological magnitude (total_persistence, the summed H0 lifetime, a direct proxy
   for diagram size)? If morphology still predicts AUC after netting out diagram
   magnitude, the association is not merely re-encoding the size of the persistence
   diagram. (This is a partial mitigation, not a break of circularity.)

Reads output/district_table.csv + output/benchmark_validation.csv.
Writes output/companion_circularity.csv. Usage: uv run python scripts/companion_and_circularity.py
"""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

MORPH = ["circuity", "mean_street_length", "orientation_entropy",
         "intersection_density", "greenspace_fragmentation"]
HEAD = ["circuity", "mean_street_length", "orientation_entropy"]


def _std_beta(model, df, term):
    return model.params[term] * df[term].std() / df["_dv"].std()


def main() -> None:
    d = pd.read_csv("output/district_table.csv")
    b = pd.read_csv("output/benchmark_validation.csv")
    df = d.merge(b, on=["city", "hex"], how="inner")
    print(f"merged n = {len(df)} over {df.city.nunique()} cities")

    # ---- #6 connectivity-loss companion outcome ----
    print("\n=== #6 Disconnection channel (degrad_unreach_auc) as companion outcome ===")
    r, p = spearmanr(df["auc"], df["degrad_unreach_auc"])
    print(f"Spearman(topological AUC, connectivity-loss AUC) = {r:+.3f} (p={p:.2g}) "
          "-> the two are nearly orthogonal/negatively related: complementary channels")
    df["_dv"] = df["degrad_unreach_auc"]
    mc = smf.ols("degrad_unreach_auc ~ " + " + ".join(MORPH + ["C(city)"]), data=df).fit()
    print("morphology -> CONNECTIVITY LOSS (std beta; sign/strength differ from AUC):")
    for t in HEAD:
        print(f"  {t:20s}: coef={mc.params[t]:+.4g}  std_beta={_std_beta(mc, df, t):+.3f}"
              f"  p={mc.pvalues[t]:.2g}")
    print(f"  R2 = {mc.rsquared:.3f}")

    # ---- #4 circularity: control for diagram magnitude (total_persistence) ----
    print("\n=== #4 Shared-origin check: morphology after netting out diagram size ===")
    df["_dv"] = df["auc"]
    m0 = smf.ols("auc ~ " + " + ".join(MORPH + ["C(city)"]), data=df).fit()
    m1 = smf.ols("auc ~ " + " + ".join(MORPH + ["total_persistence", "C(city)"]),
                 data=df).fit()
    print(f"corr(auc, total_persistence) = {df[['auc','total_persistence']].corr().iloc[0,1]:+.3f}")
    print(f"{'term':20s} {'coef0':>10s} {'coef|TP':>10s} {'p0':>8s} {'p|TP':>8s}")
    for t in HEAD:
        print(f"  {t:18s} {m0.params[t]:>+10.4g} {m1.params[t]:>+10.4g} "
              f"{m0.pvalues[t]:>8.1g} {m1.pvalues[t]:>8.1g}")
    print(f"  total_persistence coef={m1.params['total_persistence']:+.4g} "
          f"(p={m1.pvalues['total_persistence']:.2g}); R2 {m0.rsquared:.3f}->{m1.rsquared:.3f}")

    pd.DataFrame([
        {"check": "spearman_auc_vs_connloss", "value": round(float(r), 3)},
        {"check": "R2_morph_on_connloss", "value": round(float(mc.rsquared), 3)},
        {"check": "R2_auc_morph", "value": round(float(m0.rsquared), 3)},
        {"check": "R2_auc_morph_plus_totpers", "value": round(float(m1.rsquared), 3)},
    ]).to_csv("output/companion_circularity.csv", index=False)
    print("\nwrote output/companion_circularity.csv")


if __name__ == "__main__":
    main()

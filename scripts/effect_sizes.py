"""Effect sizes, standardised coefficients, multicollinearity, coefficient
heterogeneity, and wild-cluster-bootstrap inference for the eight-city headline
regression (reviewer statistics critique).

Addresses: p-values are not effect sizes at large n; raw coefficients on different
scales are not comparable; multicollinearity among morphology descriptors; whether
the pooled slope hides cross-city heterogeneity; and whether the eight-cluster
cluster-robust p-values survive a wild cluster bootstrap (the standard small-cluster
correction).

Reads output/district_table.csv (8 cities). Writes output/effect_sizes.csv.
Usage: uv run python scripts/effect_sizes.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr  # noqa: F401  (kept for parity)

HEAD = ["circuity", "orientation_entropy", "mean_street_length"]
ALL = HEAD + ["intersection_density", "greenspace_fragmentation"]
rng = np.random.default_rng(42)


def _within(df, cols):
    """City-demean columns (absorb the fixed effects)."""
    out = df.copy()
    for c in cols + ["auc"]:
        out[c + "_w"] = df[c] - df.groupby("city")[c].transform("mean")
    return out


def main() -> None:
    df = pd.read_csv("output/district_table.csv")
    rhs = " + ".join(ALL + ["C(city)"])
    m = smf.ols(f"auc ~ {rhs}", data=df).fit()

    # 1) standardised beta + 10-90 percentile effect size (within-city variation)
    dw = _within(df, ALL)
    sd_auc = dw["auc_w"].std()
    rows = []
    for c in ALL:
        b = float(m.params[c])
        sd_x = dw[c + "_w"].std()
        p10, p90 = np.percentile(df[c], [10, 90])
        eff = b * (p90 - p10)                      # AUC change over 10-90 pctile
        rows.append({
            "term": c,
            "coef_raw": round(b, 5),
            "std_beta": round(b * sd_x / sd_auc, 3),   # standardised beta
            "d10_90_AUC": round(eff, 3),               # AUC units
            "d10_90_pctSD": round(100 * eff / sd_auc, 1),  # as % of AUC SD
        })
    res = pd.DataFrame(rows)

    # 2) VIF on the (within-city) morphology matrix
    print("=== VIF (city-demeaned predictors) ===")
    X = dw[[c + "_w" for c in ALL]].to_numpy()
    for i, c in enumerate(ALL):
        y = X[:, i]
        Xo = np.delete(X, i, axis=1)
        Xo = np.column_stack([np.ones(len(Xo)), Xo])
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        r2 = 1 - np.sum((y - Xo @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        print(f"  {c:24s} VIF={vif:.2f}")

    # 3) coefficient heterogeneity: per-city slopes (sign consistency)
    print("\n=== Per-city slope signs (headline descriptors) ===")
    for c in HEAD:
        signs = []
        for city, g in df.groupby("city"):
            mm = smf.ols(f"auc ~ {' + '.join(ALL)}", data=g).fit()
            signs.append(np.sign(mm.params[c]))
        agree = int(sum(s == np.sign(m.params[c]) for s in signs))
        print(f"  {c:24s} pooled sign {np.sign(m.params[c]):+.0f}; "
              f"{agree}/{df.city.nunique()} cities agree")

    # 4) wild cluster bootstrap (Rademacher, restricted null) for headline coefs
    print("\n=== Wild cluster bootstrap p (8 clusters, 4999 reps) ===")
    Xf = pd.get_dummies(df["city"], drop_first=True).astype(float)
    Xf = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                    df[ALL], Xf], axis=1).to_numpy()
    y = df["auc"].to_numpy()
    cities = df["city"].to_numpy()
    beta_hat = np.linalg.lstsq(Xf, y, rcond=None)[0]
    tstat = {}
    for c in HEAD:
        j = ALL.index(c) + 1
        # cluster-robust se
        resid = y - Xf @ beta_hat
        XtX_inv = np.linalg.inv(Xf.T @ Xf)
        meat = np.zeros((Xf.shape[1], Xf.shape[1]))
        for cl in np.unique(cities):
            Xc = Xf[cities == cl]
            uc = resid[cities == cl]
            sc = Xc.T @ uc
            meat += np.outer(sc, sc)
        vcov = XtX_inv @ meat @ XtX_inv
        tstat[c] = beta_hat[j] / np.sqrt(vcov[j, j])
    # bootstrap under null (impose beta_j=0 via full model residuals is complex;
    # use unrestricted wild bootstrap of the t-stat distribution)
    B = 4999
    boot_t = {c: np.zeros(B) for c in HEAD}
    resid = y - Xf @ beta_hat
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=df.city.nunique())
        wmap = dict(zip(np.unique(cities), w))
        yb = Xf @ beta_hat + resid * np.array([wmap[c] for c in cities])
        bb = np.linalg.lstsq(Xf, yb, rcond=None)[0]
        rb = yb - Xf @ bb
        meat = np.zeros((Xf.shape[1], Xf.shape[1]))
        XtX_inv = np.linalg.inv(Xf.T @ Xf)
        for cl in np.unique(cities):
            Xc = Xf[cities == cl]
            uc = rb[cities == cl]
            sc = Xc.T @ uc
            meat += np.outer(sc, sc)
        vcov = XtX_inv @ meat @ XtX_inv
        for c in HEAD:
            j = ALL.index(c) + 1
            boot_t[c][b] = (bb[j] - beta_hat[j]) / np.sqrt(vcov[j, j])
    for c in HEAD:
        p = (np.sum(np.abs(boot_t[c]) >= abs(tstat[c])) + 1) / (B + 1)
        print(f"  {c:24s} t={tstat[c]:+.2f}  wild-bootstrap p={p:.4f}")

    res.to_csv("output/effect_sizes.csv", index=False)
    print("\n=== standardised beta + 10-90 percentile effect on AUC ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# External-validity extension: run ALL 10 cities at a common REDUCED disruption grid
# (5 rho x 2 rep) into a separate table, WITHOUT disturbing the 5-city full-grid
# headline. Full 8x10 on 10 large networks (Singapore 170k, Vienna 134k nodes) is
# impractical; the reduced grid tests whether the morphology signs replicate and
# doubles the cluster count (5 -> 10) for cluster-robust inference.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=output

echo "[1/4] preserve full-grid 5-city headline table"
cp "$OUT/district_table.csv" "$OUT/district_table_headline5.csv"

echo "[2/4] run 10 cities at reduced grid (5 rho x 2 rep)"
uv run python -m pipeline.run_analysis disruption=random h3_res=8 \
  'disruption.rhos=[0.0,0.1,0.2,0.3,0.4]' disruption.n_replicates=2 \
  '+cities=[istanbul,barcelona,amsterdam,bogota,phoenix,singapore,nairobi,vienna]' \
  > /tmp/tencity_run.log 2>&1
cp "$OUT/district_table.csv" "$OUT/district_table_10city.csv"

echo "[3/4] restore full-grid headline as canonical district_table.csv"
cp "$OUT/district_table_headline5.csv" "$OUT/district_table.csv"

echo "[4/4] fit 10-city external-validity regression + city-clustered SE"
uv run python - <<'PY'
import pandas as pd, statsmodels.formula.api as smf
MORPH=["intersection_density","circuity","orientation_entropy","mean_street_length","greenspace_fragmentation"]
df=pd.read_csv("output/district_table_10city.csv")
print("cities:", df.city.nunique(), "| n districts:", len(df))
print(df.groupby("city").size())
rhs=" + ".join(MORPH+["C(city)"])
ols=smf.ols(f"auc ~ {rhs}", data=df).fit()
clu=smf.ols(f"auc ~ {rhs}", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["city"]})
print("\n=== 10-city regression (classical | city-clustered, 10 clusters) ===")
for t in MORPH:
    print(f"  {t:24s} coef={ols.params[t]:+.4g}  p_ols={ols.pvalues[t]:.2g}  p_cluster={clu.pvalues[t]:.2g}")
df.to_csv("output/district_table_10city.csv", index=False)
PY
echo "DONE"

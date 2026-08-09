#!/usr/bin/env bash
# Promote to an 8-city FULL-GRID headline: run the 3 new cities at the full 8x10
# disruption grid and merge with the original 5-city full-grid table.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=output
echo "[1/3] backup 5-city full-grid headline"
cp "$OUT/district_table.csv" "$OUT/district_table_5full.csv"
echo "[2/3] run 3 new cities at FULL grid (8x10)"
uv run python -m pipeline.run_analysis disruption=random h3_res=8 \
  '+cities=[singapore,nairobi,vienna]' > /tmp/fullgrid3_run.log 2>&1
cp "$OUT/district_table.csv" "$OUT/district_table_3new_full.csv"
echo "[3/3] merge -> 8-city full-grid district_table.csv"
uv run python - <<'PY'
import pandas as pd
a=pd.read_csv("output/district_table_5full.csv")
b=pd.read_csv("output/district_table_3new_full.csv")
both=pd.concat([a,b],ignore_index=True)
both.to_csv("output/district_table.csv",index=False)
print("8-city full grid:", both.city.nunique(),"cities,",len(both),"districts")
print(both.groupby("city").size())
PY
echo "DONE_8CITY_FULLGRID"

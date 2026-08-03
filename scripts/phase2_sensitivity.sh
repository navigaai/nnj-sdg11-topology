#!/usr/bin/env bash
# Phase 2 sensitivity analyses (reviewer response):
#   - persistence-threshold sensitivity: random full grid at threshold 0.5 / 2.0
#     (headline = 1.0)
#   - betweenness-k sensitivity: targeted scenario at k = 250 / 1000 (headline 500)
# Headline artifacts are backed up and restored so the committed outputs are
# unchanged; sensitivity results are written to output/sens_*.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=output
BK=$(mktemp -d)
echo "backup headline artifacts -> $BK"
cp "$OUT/regression_random.csv" "$BK/" 2>/dev/null || true
cp "$OUT/regression_targeted.csv" "$BK/" 2>/dev/null || true
cp "$OUT/district_table.csv" "$BK/" 2>/dev/null || true
cp "$OUT/city_typology.csv" "$BK/" 2>/dev/null || true

run () {  # $1 = label ; $2 = scenario (random|targeted) ; rest = hydra overrides
  local label="$1" scenario="$2"; shift 2
  echo "=== [$label] uv run python -m pipeline.run_analysis disruption=$scenario $* ==="
  uv run python -m pipeline.run_analysis "disruption=$scenario" "$@" > "/tmp/phase2_${label}.log" 2>&1
  # Copy the scenario-specific regression output (NOT whichever file happens to
  # exist -- a previous bug copied the stale regression_random.csv for targeted).
  cp "$OUT/regression_${scenario}.csv" "$OUT/sens_${label}.csv"
  echo "    wrote $OUT/sens_${label}.csv"
}

# Threshold sensitivity (random full grid)
run "thresh0.5" random h3_res=8 persistence_threshold=0.5
run "thresh2.0" random h3_res=8 persistence_threshold=2.0

# k sensitivity (targeted). NOTE: full-run k sensitivity is very slow; the
# manuscript instead uses scripts/k_sensitivity.py (betweenness ranking stability),
# which is cheaper and answers the reviewer question directly. Kept for completeness.
run "k250"  targeted h3_res=8 +k_pivots=250
run "k1000" targeted h3_res=8 +k_pivots=1000

echo "restore headline artifacts from $BK"
cp "$BK/regression_random.csv" "$OUT/" 2>/dev/null || true
cp "$BK/regression_targeted.csv" "$OUT/" 2>/dev/null || true
cp "$BK/district_table.csv" "$OUT/" 2>/dev/null || true
cp "$BK/city_typology.csv" "$OUT/" 2>/dev/null || true
echo "=== PHASE 2 DONE ==="
for f in "$OUT"/sens_*.csv; do echo "--- $f ---"; cat "$f"; done

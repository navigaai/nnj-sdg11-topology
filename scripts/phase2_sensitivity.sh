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

run () {  # $1 = label ; rest = hydra overrides
  local label="$1"; shift
  echo "=== [$label] uv run python -m pipeline.run_analysis $* ==="
  uv run python -m pipeline.run_analysis "$@" > "/tmp/phase2_${label}.log" 2>&1
  cp "$OUT/regression_random.csv" "$OUT/sens_${label}.csv" 2>/dev/null || \
    cp "$OUT/regression_targeted.csv" "$OUT/sens_${label}.csv"
  echo "    wrote $OUT/sens_${label}.csv"
}

# Threshold sensitivity (random full grid)
run "thresh0.5" disruption=random h3_res=8 persistence_threshold=0.5
run "thresh2.0" disruption=random h3_res=8 persistence_threshold=2.0

# k sensitivity (targeted)
run "k250"  disruption=targeted h3_res=8 +k_pivots=250
run "k1000" disruption=targeted h3_res=8 +k_pivots=1000

echo "restore headline artifacts from $BK"
cp "$BK/regression_random.csv" "$OUT/" 2>/dev/null || true
cp "$BK/regression_targeted.csv" "$OUT/" 2>/dev/null || true
cp "$BK/district_table.csv" "$OUT/" 2>/dev/null || true
cp "$BK/city_typology.csv" "$OUT/" 2>/dev/null || true
echo "=== PHASE 2 DONE ==="
for f in "$OUT"/sens_*.csv; do echo "--- $f ---"; cat "$f"; done

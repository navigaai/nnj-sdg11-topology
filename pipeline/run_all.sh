#!/usr/bin/env bash
set -euo pipefail
# Prerequisite (one-time, manual): place GHS-UCDB at data/ghsl/ghs_ucdb.gpkg
CITIES=(amsterdam barcelona istanbul bogota phoenix)
# NOTE: 'hazard' scenario is disabled by default — it requires a per-city DEM
# (data/<city>/dem.tif) and hazard_nodes wiring (low_elevation_mask -> nearest nodes).
# Re-add 'hazard' only after that wiring is implemented; otherwise hazard_removal
# raises TypeError (missing hazard_nodes).
SCENARIOS=(random targeted)

# 1) Per-city baseline + city-level resilience curves (for Fig. 5)
for city in "${CITIES[@]}"; do
  uv run python pipeline/run_baseline.py city="$city"
  for sc in "${SCENARIOS[@]}"; do
    uv run python pipeline/run_disruption.py city="$city" disruption="$sc"
  done
done

# 2) District-level analysis (per-district morphology + resilience -> regression).
#    run_analysis iterates over all cities internally; one call per scenario.
for sc in "${SCENARIOS[@]}"; do
  uv run python pipeline/run_analysis.py disruption="$sc"
done
echo "all cities + scenarios complete"

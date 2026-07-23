#!/usr/bin/env bash
set -euo pipefail
# Prerequisite (one-time, manual): place GHS-UCDB at data/ghsl/ghs_ucdb.gpkg
CITIES=(amsterdam barcelona istanbul bogota phoenix)
# 'hazard' runs only for cities that have a DEM at data/<city>/dem.tif
# (e.g. Copernicus GLO-30 or SRTM clipped to the city, any CRS — it is
# reprojected internally). Cities without a DEM are skipped for hazard automatically.
SCENARIOS=(random targeted hazard)

# Invoke via `-m` (module) so the `pipeline` package is importable — running the
# scripts by path puts pipeline/ (not the repo root) on sys.path and breaks the
# `from pipeline.run_baseline import ...` imports in run_disruption / run_analysis.

# 1) Per-city baseline + city-level resilience curves (for Fig. 5)
for city in "${CITIES[@]}"; do
  uv run python -m pipeline.run_baseline city="$city"
  for sc in "${SCENARIOS[@]}"; do
    uv run python -m pipeline.run_disruption city="$city" disruption="$sc"
  done
done

# 2) District-level analysis (per-district morphology + resilience -> regression).
#    run_analysis iterates over all cities internally; one call per scenario.
for sc in "${SCENARIOS[@]}"; do
  uv run python -m pipeline.run_analysis disruption="$sc"
done
echo "all cities + scenarios complete"

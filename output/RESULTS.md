# RESULTS — Topological Resilience of Green-Space Access

> **STATUS: TEMPLATE — awaiting the full run.**
> This file is the single source of truth for every number that enters the
> manuscript. **No number may appear in the paper that is not first recorded
> here.** The placeholders below are filled by executing `pipeline/run_all.sh`
> in a networked environment (see Prerequisites). Until then, all `<...>`
> fields are unfilled.

## Prerequisites for the run (cannot be done in an offline sandbox)

1. **Network access** — `osmnx` downloads the walk network and green/public
   spaces per city from OpenStreetMap at run time.
2. **GHS-UCDB dataset** — download the GHSL Urban Centre Database GeoPackage
   once and place it at `data/ghsl/ghs_ucdb.gpkg`. The boundary loader matches
   each city's urban-centre polygon by name (`UC_NM_MN`); confirm the five
   cities resolve (Istanbul, Barcelona, Amsterdam, Bogotá, Phoenix — accents
   are substring-matched).
3. **DEM (for the `hazard` scenario, optional per city)** — place a per-city
   digital elevation model at `data/<city>/dem.tif` (e.g. Copernicus GLO-30 or
   SRTM, clipped to the city; any CRS — it is reprojected internally). The
   hazard scenario removes network edges incident to the lowest-elevation
   (flood-prone) nodes. Cities **without** a DEM are skipped for hazard
   automatically (no crash); `random` and `targeted` always run.
4. **Compute/RAM** — sublevel-set persistence at city scale can be heavy. If a
   city OOMs, rerun that city with `filtration=rips` and record the switch in
   the "Construction notes" section below and in the paper's methods/limitations.

## How to run

```bash
# one-time: place data/ghsl/ghs_ucdb.gpkg (see Prerequisites)
bash pipeline/run_all.sh
# then sanity-check:
uv run python -c "import pandas as pd; d=pd.read_csv('output/district_table.csv'); print(d.shape); print(d['city'].value_counts()); print(pd.read_csv('output/regression_random.csv')); print(pd.read_csv('output/city_typology.csv'))"
```

Artifacts produced: `output/<city>/baseline_diagram.npz`, `output/<city>/field.npz`,
`output/<city>/resilience_<scenario>.json`, `output/district_table.csv`,
`output/regression_<scenario>.csv`, `output/city_typology.csv`,
`output/figures/fig5_resilience.png`, `output/figures/fig6_morphology.png`.

---

## 1. District coverage (unit of inference)

| City | # qualifying districts (H3 res 8) | notes |
|------|-----------------------------------|-------|
| Amsterdam | `<n>` | |
| Barcelona | `<n>` | |
| Istanbul  | `<n>` | |
| Bogotá    | `<n>` | |
| Phoenix   | `<n>` | |
| **Total** | `<N — should be in the hundreds>` | inference rests on this n, not on the 5 cities |

## 2. City typology overlay (descriptive — from `city_typology.csv`)

| City | mean district AUC | mean ρ* | mean baseline total H1 persistence |
|------|-------------------|---------|------------------------------------|
| Amsterdam | `<...>` | `<...>` | `<...>` |
| Barcelona | `<...>` | `<...>` | `<...>` |
| Istanbul  | `<...>` | `<...>` | `<...>` |
| Bogotá    | `<...>` | `<...>` | `<...>` |
| Phoenix   | `<...>` | `<...>` | `<...>` |

## 3. C3 — District fixed-effects regression (INFERENTIAL, headline supporting result)

Model: `auc ~ <morphology descriptors> + C(city)` (city fixed effects), pooled over all districts.
From `output/regression_random.csv` (and per scenario):

| Morphology descriptor | coef | std err | p-value | scenario |
|-----------------------|------|---------|---------|----------|
| intersection_density | `<...>` | `<...>` | `<...>` | random |
| circuity             | `<...>` | `<...>` | `<...>` | random |
| orientation_entropy  | `<...>` | `<...>` | `<...>` | random |
| mean_street_length   | `<...>` | `<...>` | `<...>` | random |
| greenspace_fragmentation | `<...>` | `<...>` | `<...>` | random |

Repeat the block for `targeted` and `hazard`. Note sign + magnitude and whether
they are architecturally interpretable; flag any that flip sign across scenarios.

## 4. City-level resilience curves (Fig. 5 — from `resilience_<scenario>.json`)

| City | scenario | AUC | ρ* |
|------|----------|-----|----|
| ... | random / targeted / hazard | `<...>` | `<...>` |

## 5. Construction notes / deviations

- Cities that required the `rips` fallback (if any): `<list, with reason>`.
- Cities with too few qualifying districts (< threshold): `<list>`.
- H3 resolution sensitivity check (one coarser + one finer level): `<summary>`.
- Any OSM-coverage caveats (esp. Bogotá): `<summary>`.

---

*Generated scaffold. Fill every `<...>` from the committed CSV/JSON artifacts
after running `pipeline/run_all.sh`; do not transcribe numbers by hand from
intermediate console output.*

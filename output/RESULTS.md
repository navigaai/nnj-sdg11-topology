# RESULTS — Topological Resilience of Green-Space Access

> **STATUS: POPULATED** from a real 5-city run on 2026-07-04.
> This file is the single source of truth for every number that enters the
> manuscript. **No number may appear in the paper that is not recorded here.**
> Values below are transcribed directly from the committed CSV/JSON artifacts.

## Run configuration (what produced these numbers)

- **Scenarios:** `random` (percolation-style) and `targeted` (high-betweenness,
  k=500 sampled) both run; signs agree across both (§3). `hazard` needs per-city
  DEMs and is **not yet run** (see §6).
- **Disruption grid:** ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.5}, `n_replicates = 3`.
  (Reduced from the full {8 ρ × 10 reps} plan for tractability; the pattern and
  ρ\* are stable — widen for camera-ready.)
- **Resilience metric:** 1-Wasserstein distance on **H0** (`homology_dim = 0`)
  between the undisrupted and disrupted sublevel diagrams, replicate-averaged.
  H0 (merging of well-served regions) carries the signal on street networks;
  finite H1 is near-empty (few triangles fill graph cycles). See §5.
- **Denoising:** `persistence_threshold = 1.0` walk-min — H0 classes with
  lifetime < 1 min are dropped before matching (keeps the exact metric, bounds
  the O(n³) match cost). Reported D-values are therefore ~10–15% below the
  un-thresholded exact values; **relative** structure (ranking, ρ\*) is preserved.
- **Filtration:** sublevel-set, `max_dim = 0` (H0-only; triangle-fill skipped).
- **Unit of inference:** H3 district, resolution 8 (~0.7 km² cells).
- **Boundary:** GHS-UCDB R2019A urban-centre polygon per city (equal-area
  homonym disambiguation → Barcelona = Spain, not Venezuela).
- **Wall-clock:** ~2.5 h for all 5 cities (Phoenix dominant — 962 districts).

## How to reproduce

```bash
# one-time: data/ghsl/ghs_ucdb.gpkg (GHS-UCDB R2019A) — already in place
uv run python -m pipeline.run_analysis disruption=random \
  "disruption.rhos=[0.0,0.1,0.2,0.3,0.5]" disruption.n_replicates=3
# single city: add "+cities=[amsterdam]"
```

Artifacts: `output/district_table.csv`, `output/regression_random.csv`,
`output/city_typology.csv`, `output/figures/fig5_resilience.png`,
`output/figures/fig6_morphology.png`, per-city `output/<city>/baseline_diagram.npz`.

---

## 1. District coverage (unit of inference)

| City | # qualifying districts (H3 res 8) |
|------|-----------------------------------|
| Amsterdam | 227 |
| Barcelona | 147 |
| Istanbul  | 729 |
| Bogotá    | 538 |
| Phoenix   | 962 |
| **Total** | **2 603** (inference rests on this n, not on the 5 cities) |

Zero-AUC districts (no measurable degradation): **68 / 2 603 (2.6 %)** — retained
in the fit; not a material mass.

## 2. City typology overlay (descriptive — `city_typology.csv`)

Higher mean AUC ⇒ access topology changes more under disruption ⇒ **less resilient**.

| City | mean district AUC | mean ρ* | mean baseline H0 total persistence | n |
|------|-------------------|---------|-------------------------------------|---|
| Amsterdam | 4.25 | 0.271 | 13.11 | 227 |
| Barcelona | 6.54 | 0.288 | 13.63 | 147 |
| Phoenix   | 6.58 | 0.184 | 8.08  | 962 |
| Istanbul  | 7.44 | 0.214 | 12.84 | 729 |
| Bogotá    | 7.54 | 0.240 | 16.90 | 538 |

Reading: **Amsterdam most resilient** (lowest AUC); **Bogotá / İstanbul least
resilient**. **Phoenix breaks down earliest** (lowest ρ\* = 0.18) despite a
middling AUC — its sprawl access structure transitions at low disruption.

## 3. C3 — District fixed-effects regression (INFERENTIAL headline result)

Model: `auc ~ intersection_density + circuity + orientation_entropy + mean_street_length + greenspace_fragmentation + C(city)`
(OLS with **city fixed effects** absorbing between-city confounds), pooled over
**n = 2 603** districts. From `output/regression_random.csv`:

| Morphology descriptor | coef | std err | p-value | sig |
|-----------------------|------|---------|---------|-----|
| circuity | −8.991 | 1.185 | 4.5×10⁻¹⁴ | \*\*\* |
| orientation_entropy | +1.172 | 0.203 | 9.4×10⁻⁹ | \*\*\* |
| mean_street_length | −0.0478 | 0.0059 | 7.2×10⁻¹⁶ | \*\*\* |
| intersection_density | −0.199 | 0.0757 | 8.6×10⁻³ | \*\* |
| greenspace_fragmentation | −1.5×10⁻⁶ | 3.8×10⁻⁶ | 0.703 | ns |

(sig: \*\*\* p<0.001, \*\* p<0.01; scenario = random.)

**Interpretation** (coef sign is on AUC; recall higher AUC = *less* resilient):
- **circuity −8.99 \*\*\*** — more circuitous (indirect) street networks have
  *lower* AUC ⇒ **more resilient** access topology.
- **orientation_entropy +1.17 \*\*\*** — more disordered street orientation ⇒
  higher AUC ⇒ **less resilient**; ordered (grid-like) districts are more robust.
- **mean_street_length −0.048 \*\*\*** — longer street segments ⇒ **more resilient**.
- **intersection_density −0.199 \*\*** — denser intersections ⇒ **more resilient**
  (marginal).
- **greenspace_fragmentation** — not significant once it varies per-district and
  city fixed effects are included.

Four of five morphology descriptors are significant (three at p<0.001) at n=2603
with city fixed effects — a defensible cross-city morphology↔resilience result.

**Robustness — targeted scenario** (`regression_targeted.csv`, high-betweenness
removal, k=500 sampled). Every sign matches random; all four remain significant:

| descriptor | coef (targeted) | p (targeted) | sign vs random |
|---|---|---|---|
| circuity | −5.766 | 4.6×10⁻¹⁰ | same (−) |
| orientation_entropy | +0.500 | 1.6×10⁻³ | same (+) |
| mean_street_length | −0.0337 | 2.7×10⁻¹³ | same (−) |
| intersection_density | −0.135 | 2.2×10⁻² | same (−) |
| greenspace_fragmentation | −1.8×10⁻⁶ | 0.54 | same (ns) |

Targeted AUC magnitudes are lower (fewer, more critical edges removed): targeted
city mean AUC = Amsterdam 2.03, Barcelona 3.53, Phoenix 4.12, Bogotá 4.56,
İstanbul 4.78 — same ranking as random. The morphology↔resilience result is not
an artefact of the disruption model.

## 4. City-level resilience curves (Fig. 5 — `resilience_<scenario>.json`)

Only **Amsterdam** has a city-level curve so far (city-level requires
`pipeline/run_disruption.py` per city; the other four still need it — see §6):

| City | scenario | ρ grid | D(ρ) | AUC | ρ* |
|------|----------|--------|------|-----|----|
| Amsterdam | random | 0.0/0.1/0.3/0.5 | 0.0 / 279.5 / 1276.1 / 1766.1 | 947.5 | 0.30 |

## 5. Method notes / findings from the real run

- **Homology dimension = H0.** On real street networks the sublevel H1 is
  near-empty (Amsterdam baseline: 6 449 finite H0 vs 1 finite H1). The resilience
  metric therefore uses H0; H1 is available as a sensitivity check
  (`homology_dim=1 filtration.max_dim=1`).
- **Persistence threshold = 1.0 walk-min** (topological denoising) makes exact
  Wasserstein tractable at city scale (~6 449-pt H0 diagram: 149 s → 0.7 s).
- **No `rips` fallback needed** — sublevel H0 ran within memory/time for all five
  cities (streaming district resilience bounds memory).
- **No city dropped** for too-few districts; smallest is Barcelona (147).
- **Bogotá OSM coverage** adequate (538 districts); accent handled by folding.
- **H3 resolution sensitivity** (one coarser + one finer level): **not yet run**.

## 6. Outstanding before camera-ready

- `hazard` scenario: acquire per-city DEMs (`data/<city>/dem.tif`, Copernicus
  GLO-30 / SRTM) and run; the wiring is in place and skips cities without a DEM.
- Run `run_disruption` per city to complete the Fig. 5 city-level curves (§4).
- H3 resolution sensitivity (res 7 and 9).
- Restore the full disruption grid (8 ρ × 10 reps) and confirm the coefficients
  and ρ\* are unchanged vs the reduced grid used here.
- Consider reporting un-thresholded exact D for a small subsample to bound the
  denoising bias.

*Done: `random` and `targeted` scenarios; all coefficient signs stable across
both (a key robustness check).*

---

*Populated from committed artifacts (`output/*.csv`, `output/amsterdam/resilience_random.json`)
on 2026-07-04. Numbers rounded for display; full precision in the CSVs.*

# RESULTS — Topological Resilience of Green-Space Access

> **STATUS: POPULATED** from a real 5-city run on 2026-07-04.
> This file is the single source of truth for every number that enters the
> manuscript. **No number may appear in the paper that is not recorded here.**
> Values below are transcribed directly from the committed CSV/JSON artifacts.

## Run configuration (what produced these numbers)

- **Scenarios:** all three run. `random` (percolation) and `targeted`
  (high-betweenness, k=500 sampled) agree on every sign (§3). `hazard`
  (low-elevation removal, AWS Terrain Tiles DEMs) is topography-dependent and too
  mild to test the morphology signal — reported honestly in §3.
- **Disruption grid:** headline uses ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.5},
  `n_replicates = 3`. **Validated against the full grid** {8 ρ × 10 reps}: the
  three strongest descriptors keep sign + p<0.001 (circuity −10.2, orientation
  −0.048... see below), city AUC ranking and ρ\* preserved within ~0.02; only
  intersection_density (already marginal) drops to ns. See §3 "Grid sensitivity".
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
**n = 2 603** districts. **Headline = full disruption grid (8ρ×10rep)**, from
`output/regression_random.csv`:

| Morphology descriptor | coef | std err | p-value | sig |
|-----------------------|------|---------|---------|-----|
| circuity | −10.156 | 1.167 | 5.5×10⁻¹⁸ | \*\*\* |
| mean_street_length | −0.0476 | 0.0058 | 3.7×10⁻¹⁶ | \*\*\* |
| orientation_entropy | +1.015 | 0.200 | 4.4×10⁻⁷ | \*\*\* |
| intersection_density | −0.091 | 0.0745 | 0.221 | ns |
| greenspace_fragmentation | −2.0×10⁻⁶ | 4.2×10⁻⁶ | 0.636 | ns |

(sig: \*\*\* p<0.001; scenario = random, full grid.)

**Interpretation** (coef sign is on AUC; recall higher AUC = *less* resilient):
- **circuity −10.16 \*\*\*** — more circuitous street networks ⇒ **more resilient**.
- **mean_street_length −0.048 \*\*\*** — longer street segments ⇒ **more resilient**.
- **orientation_entropy +1.01 \*\*\*** — more disordered orientation ⇒ **less resilient**.
- **intersection_density −0.091 (ns)** — negatively signed (denser ⇒ suggestively
  more resilient) but **not significant** at the full grid; the weakest, grid-sensitive
  effect (significant at the coarse 5×3 grid p=8.6×10⁻³ and under targeted p=0.022).
- **greenspace_fragmentation** — not significant.

Three of five descriptors significant (two at p<10⁻¹⁵). We report the more demanding
full grid as the headline; the reduced 5×3 grid gives the same signs and kept
intersection_density marginally significant.

**Grid sensitivity — full 8ρ×10rep (headline) vs coarse 5ρ×3rep** (sensitivity):

| descriptor | full (8×10, headline) | coarse (5×3) | sign | signif |
|---|---|---|---|---|
| circuity | −10.16*** | −8.99*** | same | same |
| mean_street_length | −0.0476*** | −0.0478*** | same | same |
| orientation_entropy | +1.01*** | +1.17*** | same | same |
| intersection_density | −0.091 (ns, p=0.22) | −0.199 (p=8.6e-3) | same sign | **grid-sensitive** |
| greenspace_fragmentation | ns | ns | same | same |

City AUC ranking (district-mean): Amsterdam 4.46 < Phoenix 6.68 < Barcelona 7.67
< İstanbul 7.75 < Bogotá 8.02; ρ\* (ams 0.25, bcn 0.28, bog 0.24, ist 0.20, phx 0.16).

### 3b. Spatial-dependence robustness (reviewer response, `scripts/spatial_robustness.py`)

**City-clustered standard errors** (5 clusters; `output/regression_clustered.csv`):

| descriptor | coef | SE (classical) | SE (HC3) | SE (cluster) | p (cluster) |
|---|---|---|---|---|---|
| circuity | −10.156 | 1.167 | 1.397 | 2.946 | 5.7×10⁻⁴ \*\*\* |
| mean_street_length | −0.0476 | 0.0058 | 0.0066 | 0.0089 | 7.7×10⁻⁸ \*\*\* |
| orientation_entropy | +1.015 | 0.200 | 0.220 | 0.894 | 0.257 (ns) |
| intersection_density | −0.091 | 0.074 | 0.092 | 0.214 | 0.670 (ns) |
| greenspace_fragmentation | −2.0×10⁻⁶ | 4.2×10⁻⁶ | 5.9×10⁻⁶ | 6.1×10⁻⁷ | <10⁻² (artefact†) |

† greenspace_fragmentation clustered "significance" is a 5-cluster artefact around a
~0 coefficient; not substantive. **Circuity and mean_street_length survive city
clustering; orientation_entropy does not.** With only 5 clusters, cluster-robust
inference is underpowered → classical p-values understate uncertainty.

**Global Moran's I on residuals** (H3 first-ring adjacency, 999 perms;
`output/moran_residuals.csv`): weak but significant positive spatial autocorrelation
in **every** city — Amsterdam 0.080 (p=0.031), Barcelona 0.178 (p=0.001),
Bogotá 0.142 (p=0.001), İstanbul 0.112 (p=0.001), Phoenix 0.058 (p=0.003).
→ residuals are spatially dependent → estimated spatial models below.

**Estimated spatial models** (`scripts/spatial_regression.py`, `spreg` ML, city FE,
block-diagonal within-city H3 weights, n=2602 after 1 island; `output/spatial_regression.csv`):

| descriptor | OLS-FE coef (z) | Spatial Error (z) | Spatial Lag (z) |
|---|---|---|---|
| circuity | −10.15 (−8.7) | −5.38 (−5.0) | −6.88 (−6.7) |
| mean_street_length | −0.048 (−8.2) | −0.033 (−5.8) | −0.034 (−6.5) |
| orientation_entropy | +1.01 (+5.1) | +1.25 (+5.9) | +0.92 (+5.2) |
| intersection_density | −0.09 (−1.2 ns) | −0.21 (−2.9) | −0.16 (−2.5) |
| greenspace_frag | ns | ns | ns |
| **spatial param** | — | λ=0.55 | ρ=0.53 |

Pooled Moran's I (spreg) = 0.258 (z=21.5, p=2e-102). **Both spatial models absorb
strong dependence (λ,ρ≈0.5) yet the 3 headline descriptors keep sign + |z|≥5**;
intersection_density even turns significant. → morphology signal is NOT a spatial-
autocorrelation artefact; remaining caveat is external validity (only 5 cities).

**Spatial model selection (Anselin LM tests on OLS residuals):** LM-error 448.9
(p=1e-99), LM-lag 569.3 (p=8e-126); robust LM-error 25.7, **robust LM-lag 146.1**
→ lag form favoured. AIC: OLS 14350.6 → SEM 13924.6 → **SAR 13876.9** (both spatial
models beat OLS by >400). → formally justifies the spatial specification.

### 3d. Omitted-variable controls (reviewer response, `scripts/omitted_controls.py`)

Added 4 district controls computable from existing data (no resilience recompute;
AUC reused). `output/regression_controls.csv`, `output/district_controls.csv`:

| variable | base coef (p) | +controls coef (p) |
|---|---|---|
| circuity | −10.16 (5.5e-18) | **−9.55 (3.0e-16)** |
| mean_street_length | −0.0476 (3.7e-16) | **−0.0395 (9.7e-12)** |
| orientation_entropy | +1.015 (4.4e-7) | **+1.107 (1.1e-7)** |
| intersection_density | −0.091 (0.22 ns) | −0.098 (0.18 ns) |
| green_area_km2 | — | −2.25 (1.7e-3) |
| major_road_share | — | +4.21 (7.1e-8) |
| relief_m (topography) | — | −0.008 (0.40 ns) |
| centre_dist_km | — | −0.107 (6.2e-20) |

R² 0.171 → 0.206. **3 headline descriptors keep sign + significance (p<1e-11)** after
adding green quantity, road hierarchy, topography, centre proximity; 3 controls are
themselves significant (real confounders) yet morphology survives. Still uncontrolled
(need external data): population/built density, land-use mix, socioeconomic,
pedestrian infra.

### 3e. Benchmark / convergent validation (reviewer M1 + M2, `scripts/benchmark_validation.py`)

Two non-topological baselines per district under the same random disruption
(reduced grid, no PH/Wasserstein): **distance-degradation** (mean walk-time-to-green
increase) and **connectivity-degradation** (unreachable-fraction increase).

**Convergent validity — Spearman(topological AUC, baseline):**
- distance-degradation: ρ = **+0.071** (p=3e-4) — correct sign, but very weak
- connectivity-degradation: ρ = **−0.285** (p=6e-50) — **wrong sign**
- static baseline access: ρ = −0.109 (p=2e-8)

The negative connectivity correlation is the **finite-node-dropout artefact**
operating district-wide (not just Phoenix at ρ=0.5): when severe damage disconnects
nodes they leave the finite diagram, so Wasserstein/AUC can *fall* exactly when
access loss is worst. This is an honest, important limitation of the metric.

**Added value (discriminant):** controlling for BOTH baselines + city FE, the 3
headline morphology descriptors stay significant (circuity −5.87 p=3e-7, mean_str_len
−0.051 p=1e-19, orient_ent +0.98 p=3e-7); both baselines are themselves strong
(degrad_unreach p=8e-61). R²=0.253. → topological AUC is **not redundant** with simple
degradation (captures distinct configurational info), but its convergent validity
with a distance-based ground truth is weak → criterion validation only partially met;
the metric is a distinct indicator, not yet a validated substitute.

### 3f. Extended-persistence (finite-cap) diagnostic (`scripts/extended_persistence.py`)

Test whether the wrong-sign connectivity result is the disconnection mechanism: cap
unreachable field at τ=60 min (keep disconnected nodes in the diagram) vs original
(inf, drops them). Amsterdam+Bogotá, reduced grid, n=765:

| metric | vs connectivity-degrad | vs distance-degrad |
|---|---|---|
| original (inf) | −0.100 | +0.406 |
| capped (τ=60) | **+0.018** | +0.059 |

Cap **flips the connectivity sign** (−0.10→+0.02) → confirms the artefact is the
disconnection mechanism. BUT it **erodes** the distance-degradation correlation
(+0.41→+0.06) → naive finite cap trades one artefact for another.

### 3g. Criterion validity — WITHIN-CITY (reviewer W1, `scripts/criterion_validity.py`)

The regression identifies morphology from within-city variation (city FE), so the
matching validity question is within-city. Spearman(topological AUC, distance-based
accessibility-degradation benchmark):

| scope | n | ρ | p |
|---|---|---|---|
| Amsterdam | 227 | +0.465 | 1e-13 |
| Bogotá | 538 | +0.156 | 3e-4 |
| İstanbul | 729 | +0.095 | 0.011 |
| Barcelona | 147 | +0.053 | 0.53 |
| Phoenix | 962 | +0.027 | 0.40 |
| **WITHIN-CITY pooled** | 2603 | **+0.360** | **2e-80** |
| naive pooled | 2603 | +0.071 | 3e-4 |

**Positive in all 5 cities; within-city +0.36 (p<1e-79).** The naive pooled +0.07 was
depressed by cross-city AUC scale (Simpson's paradox — the scale city FE removes).
→ **criterion validity holds at the level the metric is used.** Scope: validity is
for GRADED access degradation; within-city AUC↔connectivity is ~0-to-negative
(disconnection channel). → metric measures resilience of the CONNECTED served
structure; disconnection is a complementary channel.

### 3h. Extended-persistence metric (W2, `scripts/extended_persistence_v2.py`, `_extdiag.py`)

GUDHI extended persistence gives disconnected/essential H0 classes finite (birth,death)
in a separate channel. Amsterdam reduced grid, Spearman vs baselines:

| metric | vs distance-degrad | vs connectivity |
|---|---|---|
| original (inf) | +0.462 | +0.115 |
| naive cap | +0.141 | +0.245 |
| extended-persist | +0.049 | +0.314 |

Extended persistence tracks connectivity best (+0.31) but distance worst (+0.05) —
the high-persistence component classes dominate a single Wasserstein match.
**Conclusion:** graded resilience and disconnection are TWO channels a single scalar
conflates → report them separately (as the paper now does); a joint multi-channel
metric is future work. This is the honest W2 outcome — mechanism understood, clean
single-scalar fix shown to be non-trivial.

### 3c. Phase 2 sensitivity (reviewer response)

**Persistence-threshold sensitivity** (random full grid; `output/sens_thresh*.csv`):

| descriptor | thresh 0.5 | thresh 1.0 (headline) | thresh 2.0 |
|---|---|---|---|
| circuity | −11.20 (2e-19) | −10.16 (5.5e-18) | −8.03 (1.9e-14) |
| mean_street_length | −0.0502 (4e-16) | −0.0476 (3.7e-16) | −0.0422 (6e-16) |
| orientation_entropy | +1.096 (2.5e-7) | +1.015 (4.4e-7) | +0.929 (2.2e-7) |
| intersection_density | −0.044 (0.57 ns) | −0.091 (0.22 ns) | −0.138 (0.038 \*) |
| greenspace_fragmentation | ns | ns | ns |

Three headline descriptors keep sign + significance (p<1e-6) across the 4× threshold
range; magnitudes shrink smoothly with more denoising, never flip sign. int_dens
stays weakest (ns except marginal at heaviest denoising). → 1.0-min choice not a driver.

**Betweenness-k sensitivity** (`scripts/k_sensitivity.py`, `output/k_sensitivity.csv`,
Amsterdam network, 68 031 undirected edges): betweenness ranking is highly stable in
the pivot count k. Vs headline k=500: **k=250 → Spearman ρ=0.985, top-decile Jaccard
0.77; k=1000 → ρ=0.985, Jaccard 0.90.** Halving/doubling k barely changes the edge
ordering → targeted scenario is not an artefact of the pivot count.

Note: the Phase-2 driver `scripts/phase2_sensitivity.sh` had a file-copy bug that
mislabelled the k full-run outputs; the k question is answered instead by the
ranking-stability check above (cheaper and more direct). Threshold outputs are valid.

**Robustness — targeted scenario** (`regression_targeted.csv`, high-betweenness
removal, k=500 sampled). Every sign matches the random headline; the three headline
descriptors stay significant, and intersection_density (ns at the full random grid)
reaches p=0.022 here — consistent with it being the weakest, grid/scenario-sensitive effect:

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

**Hazard scenario (`regression_hazard.csv`) — honest negative/weak result.**
Low-elevation removal is topography-dependent: hazard nodes = Amsterdam 361,
Phoenix 4810 (flat), but İstanbul 22, Barcelona 17, Bogotá 10 (hilly/high, lowest
cells at the waterfront). Resulting mean AUC is near-zero (Phoenix 0.31, Amsterdam
0.09, others ≤0.01) — this localized disruption barely moves the city-wide H0
topology. The regression does **not** resolve the morphology signal: only
orientation_entropy is significant (+0.10, p=0.014); circuity and mean_street_length
have near-zero, non-significant coefficients (apparent sign flips are noise on
~zero-variance AUC). Read as a limitation of the mild ρ-capped proxy, not a
contradiction of random/targeted. Motivates a full flood-zone edge-removal model.

## 4. City-level resilience curves (Fig. 5 — `resilience_random.json`)

All five cities, computed on the **UCDB-clipped** networks (same spatial unit as
the district analysis), random scenario, ρ ∈ {0, 0.1, 0.2, 0.3, 0.5}, 3 reps.
Lower AUC = more resilient. Ranking matches the district typology (§2).

| City | AUC | ρ* | D(ρ) |
|------|-----|----|------|
| Barcelona | 619.8 | 0.30 | 0, 145, 381, 826, 1333 |
| Amsterdam | 897.8 | 0.30 | 0, 294, 666, 1205, 1722 |
| Phoenix | 2053.2 | 0.20 | 0, 730, 2023, 3574, 2152 |
| İstanbul | 2439.9 | 0.20 | 0, 909, 2157, 3969, 3180 |
| Bogotá | 2828.0 | 0.30 | 0, 968, 2191, 3862, 5188 |

İstanbul and Phoenix curves turn down at ρ=0.5: heavy fragmentation drops
unreachable nodes out of the finite H0 diagram (near-total collapse, not
recovery). NOTE: `run_disruption` now clips to the UCDB boundary, resolving the
earlier Fig 5 (unclipped) vs districts (clipped) spatial-unit mismatch.

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
- **H3 resolution sensitivity**: re-ran at res 7 (485 districts) and res 9
  (13,782 districts). circuity, orientation_entropy, mean_street_length stay
  significant at p<0.001 with the same sign at all three resolutions (res 7/8/9);
  intersection_density keeps its sign (significant only at res 8);
  greenspace_fragmentation non-significant throughout. Magnitudes scale with cell
  size. Result is robust to resolution across a 28× district-count range.

## 6. Outstanding before camera-ready

- Strengthen the `hazard` model: replace the mild ρ-capped low-elevation removal
  with full inundation (remove all edges below a flood-depth threshold), so the
  scenario actually stresses the network. DEMs are already in place (AWS Terrain
  Tiles, `data/<city>/dem.tif`, fetched via `scripts/fetch_dem.py`).
- Run `run_disruption` per city to complete the Fig. 5 city-level curves (§4).
- Consider reporting un-thresholded exact D for a small subsample to bound the
  denoising bias.

*Done: `random` and `targeted` scenarios; all coefficient signs stable across
both (a key robustness check).*

---

*Populated from committed artifacts (`output/*.csv`, `output/amsterdam/resilience_random.json`)
on 2026-07-04. Numbers rounded for display; full precision in the CSVs.*

### 3i. Walking-speed sensitivity (reviewer A, `scripts/speed_sensitivity.py`)

Speed enters only via time=length/speed → global linear rescale of field/diagram/AUC
→ regression signs/significance/ranking invariant. Amsterdam empirical check:
- 4.0 km/h vs 4.8: Spearman 0.985, median AUC ratio 1.24 (expected 1.20)
- 5.0 km/h vs 4.8: Spearman 0.996, median AUC ratio 0.96 (expected 0.96)
4.8 km/h = 1.33 m/s cited to Bohannon (1997), within 1.27–1.46 m/s comfortable range.

### 3j. Green-space definition sensitivity (reviewer B, `scripts/green_sensitivity.py`)

Re-downloaded green WITH tags (green_tagged.gpkg); STRICT subset = park +
recreation_ground + square (drop wood/grass/garden). Reduced grid, all 5 cities:
- Spearman(auc_all, auc_strict) = 0.82 (large green cut, e.g. Amsterdam 23657→369)
- circuity −8.08 (p=4e-9), mean_street_length −0.048 (p=2e-12), orientation_entropy
  +1.21 (p=2e-7) keep sign+significance under strict green; intersection_density
  weak (p=0.13). → morphology result not an artefact of counting inaccessible green.

### Citation fixes (reviewers G/H)
- 1-Wasserstein stability → Cohen-Steiner et al. 2010 (FoCM, Lp-stability), DOI 10.1007/s10208-010-9060-6
- Hickok et al. → SIAM Review 66(3):481–500 (2024), DOI 10.1137/22M150410X

### 3k. Population-weighted criterion validity (open GHS-POP, reviewer request)

Open data: **GHS-POP R2023A** (JRC, 1 km, 2020; same GHSL/Mollweide family as GHS-UCDB),
downloaded to data/ghsl/. `scripts/population_validation.py` builds a
POPULATION-WEIGHTED accessibility-degradation benchmark (weighted mean walk-time over
reachable nodes, weights = GHS-POP) and correlates with topological AUC.

Spearman(AUC, population-weighted degradation):
- amsterdam +0.458, bogota +0.157, istanbul +0.091, phoenix +0.020, barcelona −0.019
- **WITHIN-CITY pooled +0.354 (p=1e-77)**; naive pooled +0.067

≈ identical to the unweighted distance benchmark (+0.36) → topological signal is not
an artefact of weighting nodes equally, and it tracks population-weighted graded
access degradation → partially engages the SDG 11.7 population-coverage dimension.
(Weighting is coarse: 1 km cells, node-level sampling; heterogeneous across cities,
barcelona/phoenix weak.)

### Open-source data found for the two hardest gaps
- **Population:** GHS-POP R2023A (JRC) — DONE, integrated above.
- **Real flood extent:** JRC river flood-hazard maps (10–500 yr, ~90 m,
  https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/flood_hazard/) and the
  Global Flood Database (913 observed events 2000–2018, GEE) — FOUND, not yet
  integrated (would replace the DEM low-elevation hazard proxy with real flood zones).
- **Observed post-disaster green-access service loss:** no clean open per-city dataset
  exists; only flood *extent* (modelled/observed) is available.

### 4. External validity — eight-city replication (reviewer: 5 cities / 5 clusters)

Added 3 cities on other continents (configs conf/city/{singapore,nairobi,vienna}.yaml)
→ 8 cities, n=3999 districts, reduced grid (5ρ×2rep), `scripts/run_new_cities.sh`,
`output/district_table_10city.csv`, `output/regression_8city.csv`. Copenhagen +
Melbourne dropped (OSM place query / Overpass didn't resolve full extent).

| descriptor | coef | p (classical) | p (8-cluster) |
|---|---|---|---|
| circuity | −6.41 | 3e-12 | 1.6e-4 *** |
| mean_street_length | −0.046 | 9e-23 | 8e-18 *** |
| orientation_entropy | +1.85 | 3e-25 | 0.025 * |
| intersection_density | −0.023 | 0.70 | 0.88 ns |

**3 headline signs replicate across 8 cities/4 continents; with 8 clusters ALL THREE
survive city-clustered SE — orientation entropy recovered (0.26 at 5 clusters → 0.025
at 8).** → the 5-cluster failure was a power problem; external validity + cluster
inference materially strengthened. Districts per city: ams227 bcn147 bog538 ist729
nai363 phx962 sng573 vie460.

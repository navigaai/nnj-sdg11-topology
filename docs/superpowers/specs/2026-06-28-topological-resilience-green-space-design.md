# Design Spec — Topological Resilience of Green-Space Access

**Date:** 2026-06-28
**Target venue:** *Nexus Network Journal* (Springer) — Topical Collection **"Math for SDG 11 — Sustainable Cities and Communities"** (Guest editors: Karolina Ostrowska-Wawryniuk, Marcin Strzała). Collection: https://link.springer.com/collections/gdejjadigc
**Paper type:** Original research article (architecture & mathematics).
**Authorship:** Single-author, user as lead.

---

## 1. Working title

*The Topology of Reachable Nature: Persistent Homology and the Resilience of Green-Space Access in Cities of Contrasting Morphology.*

(Alternatives to test later: "Persistent Deserts: A Topological Measure of Green-Space Access Resilience for SDG 11"; "When Access Breaks: Topological Resilience of Public Green Space under Urban Disruption".)

## 2. Motivation & venue fit

SDG 11.7 calls for *universal access to safe, inclusive, accessible green and public spaces*. Access is usually measured with GIS catchment / proximity methods that report a static snapshot. Two gaps matter:

1. Static accessibility says nothing about **resilience** — how access degrades when the network is disrupted (flood, earthquake, road/bridge closure).
2. Existing accessibility work rarely connects the *shape* of access to **urban morphology**, which is precisely the architecture-mathematics question NNJ rewards.

Persistent homology (PH) gives a principled, multi-scale, coordinate-free language for "holes in coverage." The contribution is to move from *detecting* deserts (already done for other amenities) to *measuring the resilience of the access topology* and *linking it to urban form*.

## 3. Positioning against prior art (honest novelty boundary)

PH for resource accessibility already exists and must be cited and differentiated:

- Persistent homology for resource coverage / polling sites — arXiv:2206.04834
- Congestion barcodes (topology of urban congestion) — arXiv:1707.08557
- Healthcare resource accessibility via PH — arXiv:2512.12011
- Displacement / gentrification via PH — arXiv:2512.10753
- TDA of spatial systems (review) — arXiv:2104.00720

**What is NOT novel:** using PH to find accessibility "holes" for an amenity.
**What IS novel (our contribution):**
1. A **topological resilience metric** — persistence-diagram distance (bottleneck / Wasserstein) between pre- and post-disruption diagrams as a function of disruption intensity, with a critical transition ρ\* (linking to percolation).
2. **First** application to **SDG 11.7 green/public space**, framed for an **architecture-mathematics** audience.
3. An empirical **morphology ↔ resilience** map across cities of contrasting form.

## 4. Research questions

- **RQ1.** Can population-weighted persistent homology characterize equitable green/public-space access as a topological signature (deserts = persistent voids)?
- **RQ2 (core, the mathematics).** How does the access topology degrade under spatial disruption, measured as persistence-diagram distance vs. disruption intensity, and is there a critical transition ρ\* (percolation link)?
- **RQ3 (NNJ bridge, district-level).** Across *intra-urban districts* (the unit of analysis), how do urban-morphology descriptors (intersection density, circuity, orientation entropy, block size, green-space fragmentation) relate to topological resilience? Cities serve as a typology layer over the district-level relationship, not as the statistical sample.

## 5. Contributions (restated, ordered for the "Math for SDG 11" call)

C1 (headline, the mathematics) — A **topological resilience metric** for accessibility: the persistence-diagram-distance resilience curve `D(ρ)` (bottleneck / Wasserstein), its AUC, and the critical transition ρ\* linked to percolation. This is the contribution the special issue rewards directly.
C2 (framing) — First TDA treatment of SDG 11.7 green/public-space access in an architecture-mathematics setting.
C3 (supporting, district-level) — A statistically defensible **morphology↔resilience** relationship estimated across intra-urban districts (n in the hundreds), with the five cities providing a typological reading and design implications.

## 6. Methodology

All steps use open data and open-source Python.

### 6.1 Data acquisition
- **Urban boundary (comparability):** use the **GHSL Urban Centre Database (GHS-UCDB)** built-up urban-centre polygon for each city as the analysis boundary, rather than administrative limits. This gives a consistent, reproducible, morphology-agnostic extent so cross-city numbers are comparable (avoids the Phoenix-metro vs. Amsterdam-municipality mismatch).
- **Walk network:** OpenStreetMap via `osmnx` (`network_type="walk"`), clipped to the GHS-UCDB boundary.
- **Green/public space:** OSM tags `leisure=park|garden|recreation_ground`, `landuse=grass|recreation_ground`, `place=square`, `natural=wood` clipped to the urban boundary; access points = boundary nodes / entrances / centroids snapped to the walk network.
- **Population:** GHSL (GHS-POP) or WorldPop gridded population (open, global) → demand points.
- **Elevation / hazard proxy:** Copernicus DEM or SRTM for flood/seismic-zone disruption.

### 6.2 Accessibility field
- Snap population cells to nearest network node.
- Network walking travel time (assume ~4.8 km/h) from each population node to nearest green-space access point ⇒ scalar field `f: V → ℝ`.

### 6.3 Persistent homology
Two complementary constructions (report both, cross-check):
- **(a) Sublevel-set filtration** of `f` on the spatial domain: H0 tracks merging of well-served regions; H1 captures enclosed access deserts.
- **(b) Network-metric Vietoris–Rips / weighted alpha** over access points using network distance: coverage holes appear as H1 classes whose **death time = travel time required to fill the hole** (cf. polling-sites construction).
- **Population weighting:** weight/threshold by population so persistence reflects people affected (equity dimension), not empty land.
- Output: persistence diagrams / barcodes per city.

### 6.4 Disruption model (novel core)
Three comparable scenarios across all cities:
1. **Random edge removal** (percolation-style) at increasing fraction ρ.
2. **Targeted removal** of high-betweenness bridges/links.
3. **Hazard-zone removal** — drop network elements in low-elevation (flood) or high-risk (seismic) zones from DEM.

For each ρ: recompute `f`, recompute persistence diagram, compute **bottleneck distance d_B** and **Wasserstein distance W_p** to the undisrupted diagram ⇒ **resilience curve** `D(ρ)`. Summaries: area under curve (AUC) and **critical ρ\*** where new persistent deserts are born (topological transition; connects to percolation phase transition).

### 6.5 District-level analysis and cross-city comparison
- **Unit of analysis = district.** Tile each city's urban-centre boundary into districts using a uniform **H3 hexagonal grid** (fixed resolution, e.g. res 8 ≈ 0.7 km² cells; report sensitivity to one coarser/finer level). Each hex with sufficient network coverage is one observation.
- For each hex: compute its local resilience summaries (AUC, ρ\*, baseline total H1 persistence) from the disruption pipeline restricted to that hex's catchment, and its local morphology descriptors (`osmnx` stats: intersection density, circuity, orientation entropy, mean block size; green-space fragmentation).
- **Statistics:** regress / correlate resilience against morphology across the **pooled district sample** (n in the hundreds), with city as a grouping factor (mixed-effects or city fixed effects) to absorb between-city confounds. This is the architecture-mathematics bridge and the statistically defensible form of C3.
- **City typology layer:** summarize the five cities (baseline deserts, mean district resilience, resilience-curve shape) as an interpretive overlay on the district-level relationship — not as the n=5 sample.
- **Selection rationale (state explicitly in the paper):** the five cities are a *purposive maximum-variance* sample chosen to span street-network morphology (grid ↔ organic ↔ sprawl), planning regime, hazard exposure (seismic/flood), and the Global North–South divide relevant to SDG 11.7; they are deep-dive exemplars, while inference rests on districts.

### 6.6 Stack
`osmnx`, `networkx`, `geopandas`, `shapely`, `rasterio` (population/DEM), `h3` (district hex tiling), `gudhi` / `ripser` / `persim` (PH + diagram distances), `POT` (Wasserstein), `statsmodels` (mixed-effects / fixed-effects regression), `matplotlib`. Managed with `uv`. Random seeds fixed; configs via Hydra/OmegaConf where useful.

## 7. Case-study cities (typology layer — purposive maximum-variance sample)

> These five cities are deep-dive exemplars spanning the morphology/planning/hazard/North–South axes (see §6.5 selection rationale). Statistical inference for RQ3/C3 rests on the pooled **district** sample, not on these five points.


| City | Morphology character | Role |
|------|----------------------|------|
| İstanbul | Organic, hilly, polycentric, seismic | Turkey anchor; resilience story |
| Barcelona | Cerdà grid + superblocks | Planned-grid + SDG 11 intervention |
| Amsterdam | Compact organic, canal | Global-North walkable benchmark |
| Bogotá | Dense, informal+formal mix | Global-South equity contrast |
| Phoenix | Low-density sprawl, grid | Sprawl / car-oriented contrast |

## 8. Figures
1. Pipeline schematic (data → field → filtration → diagram → disruption → resilience curve).
2. Per-city green-space + walk-network maps.
3. Accessibility-field heatmaps.
4. Per-city persistence barcodes / diagrams.
5. Resilience curves `D(ρ)` across cities and scenarios.
6. Morphology-vs-resilience scatter (the headline figure).

## 9. Paper structure

1. Introduction — SDG 11.7, access vs. resilience, why topology.
2. Related work — TDA in urban analysis; green-space accessibility & equity; urban resilience. State the gap.
3. Mathematical background — simplicial complexes, filtrations, persistent homology, bottleneck/Wasserstein distances (clear exposition for the NNJ audience).
4. Methodology — pipeline + resilience metric.
5. Case studies — five cities, data, parameters.
6. Results & comparison.
7. Discussion — design/morphology implications, SDG 11 policy relevance, limitations.
8. Conclusion & future work.

## 10. Limitations to pre-empt (address in §7)
- OSM completeness varies by city (validate coverage, especially Bogotá).
- Green-space access-point definition sensitivity (entrances vs. centroids).
- Population-grid resolution and snapping error.
- Disruption-model assumptions (idealized failures, not full hazard simulation).
- PH computational cost at city scale (mitigate via sparsification / cubical complexes on rasterized field).

## 11. Logistics
- Project root: `~/Naviga_academic/nnj-sdg11-topology/` (git).
- Spec: this file under `docs/superpowers/specs/`.
- Next step after spec approval: implementation plan via the writing-plans skill, then phased execution (data → PH baseline → disruption/resilience → comparison → figures → manuscript).

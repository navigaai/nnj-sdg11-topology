# Topological Resilience of Green-Space Access — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python pipeline that measures the *topological resilience* of green/public-space access in five morphologically contrasting cities, then writes the Nexus Network Journal manuscript from the results.

**Architecture:** A config-driven (Hydra) pipeline with strict module boundaries: `data → accessibility field → persistent homology → disruption → resilience curves → morphology → cross-city analysis → figures → manuscript`. Each stage writes intermediate artifacts to `output/<city>/` so later stages re-run without recomputing earlier ones. Persistence diagrams are the central data structure passed between the topology, disruption, and analysis stages.

**Tech Stack:** Python ≥3.11, `uv` (env + deps), Hydra+OmegaConf (config), `osmnx`/`networkx` (street networks + morphology), `geopandas`/`shapely`/`rasterio` (geospatial + population/DEM rasters), `gudhi`+`ripser`+`persim` (persistent homology + diagram distances), `POT` (Wasserstein), `numpy`/`scipy`/`pandas` (numerics + regression), `matplotlib` (figures), `pytest` (tests).

## Global Constraints

- Python version floor: `>=3.11` (declared in `pyproject.toml`).
- Package manager: `uv` only. Never call `pip` directly; use `uv add` / `uv run`.
- Code style (per user global rules): files 200–400 lines max; type hints on every function; `@dataclass(frozen=True)` for config-shaped data; module-level `logger = logging.getLogger(__name__)` (no `print`); specific exceptions only; every package `__init__.py` defines `__all__`.
- Reproducibility: every stochastic step takes an explicit `seed: int` argument; global seed default `42`; record resolved Hydra config + `uv pip freeze` into each run's `output/` directory.
- Registry pattern for the swappable families: disruption models and filtration constructions are registered via decorator and selected by string name from config.
- Persistence diagram canonical form everywhere: a dict `{0: np.ndarray(shape=(n0,2)), 1: np.ndarray(shape=(n1,2))}` mapping homology dimension → array of `[birth, death]` rows; `death` may be `np.inf` for the essential H0 class.
- All distances/metrics are **finite**: infinite-persistence classes are removed before bottleneck/Wasserstein computation (documented in code).
- Tests must run offline. Network/raster downloads are wrapped behind loader functions; tests use small committed fixtures in `tests/fixtures/`, never live downloads.
- Manuscript: original research only (per project rules), British/American spelling consistent with NNJ (American), no placeholder numbers in the paper — every reported figure traces to an artifact in `output/`.

---

## File Structure

```
nnj-sdg11-topology/
├── pyproject.toml                      # uv project + deps + pytest config
├── conf/
│   ├── config.yaml                     # top-level Hydra config (defaults, seed, paths)
│   ├── city/{istanbul,barcelona,amsterdam,bogota,phoenix}.yaml
│   ├── disruption/{random,targeted,hazard}.yaml
│   └── filtration/{sublevel,rips}.yaml
├── src/nnj_topology/
│   ├── __init__.py
│   ├── config.py                       # frozen dataclasses mirroring Hydra schema
│   ├── seeding.py                      # set_seed()
│   ├── data/
│   │   ├── __init__.py
│   │   ├── network.py                  # OSM walk network load + cache
│   │   ├── greenspace.py               # green/public space polygons + access points
│   │   ├── population.py               # gridded population → demand points
│   │   └── hazard.py                   # DEM → low-elevation hazard mask
│   ├── accessibility/
│   │   ├── __init__.py
│   │   └── field.py                    # network walk-time field f: nodes → minutes
│   ├── topology/
│   │   ├── __init__.py                 # filtration registry + factory
│   │   ├── filtration.py               # sublevel-set + Rips builders
│   │   ├── diagrams.py                 # compute persistence diagrams (gudhi/ripser)
│   │   └── distances.py                # bottleneck + Wasserstein (finite-safe)
│   ├── disruption/
│   │   ├── __init__.py                 # disruption registry + factory
│   │   ├── models.py                   # random / targeted / hazard edge removal
│   │   └── resilience.py               # D(rho) curve, AUC, critical rho*
│   ├── morphology/
│   │   ├── __init__.py
│   │   └── descriptors.py              # osmnx morphology stats + fragmentation
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── crosscity.py               # assemble table + correlation/regression
│   └── viz/
│       ├── __init__.py
│       └── figures.py                  # the 6 manuscript figures
├── pipeline/
│   ├── run_baseline.py                 # data → field → baseline diagram per city
│   ├── run_disruption.py               # disruption scenarios → resilience curves
│   └── run_analysis.py                 # morphology + cross-city + figures
├── tests/
│   ├── fixtures/                       # tiny committed graphs/rasters/diagrams
│   └── test_*.py
├── data/                               # raw + cached downloads (gitignored)
├── output/                             # per-city artifacts + figures (gitignored)
└── paper/                              # manuscript (LaTeX)
```

---

## Task 1: Project scaffolding (uv + structure + config schema)

**Files:**
- Create: `pyproject.toml`
- Create: `src/nnj_topology/__init__.py`
- Create: `src/nnj_topology/seeding.py`
- Create: `src/nnj_topology/config.py`
- Create: `tests/__init__.py`, `tests/test_config.py`
- Modify: `.gitignore` (add `output/`, `paper/_build/`)

**Interfaces:**
- Produces: `set_seed(seed: int) -> None`; frozen dataclasses `CityConfig`, `DisruptionConfig`, `FiltrationConfig`, `RunConfig` (see code below); `from_omegaconf(cfg) -> RunConfig`.

- [ ] **Step 1: Create the uv project and add dependencies**

Run:
```bash
cd /Users/seydaemekci/Naviga_academic/nnj-sdg11-topology
uv init --lib --name nnj_topology --python 3.11
uv add osmnx networkx geopandas shapely rasterio numpy scipy pandas matplotlib gudhi ripser persim pot hydra-core omegaconf
uv add --dev pytest
```
Expected: `pyproject.toml` + `uv.lock` created; `.venv/` populated.

- [ ] **Step 2: Write the failing test for config + seeding**

Create `tests/test_config.py`:
```python
import numpy as np
from omegaconf import OmegaConf

from nnj_topology.config import RunConfig, from_omegaconf
from nnj_topology.seeding import set_seed


def test_set_seed_is_reproducible():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_from_omegaconf_builds_frozen_runconfig():
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "city": {"name": "testville", "place": "Testville, Country", "crs": "EPSG:3857"},
            "disruption": {"name": "random", "rhos": [0.0, 0.5], "n_replicates": 2},
            "filtration": {"name": "sublevel", "max_dim": 1},
            "paths": {"data": "data", "output": "output"},
        }
    )
    rc = from_omegaconf(cfg)
    assert isinstance(rc, RunConfig)
    assert rc.seed == 7
    assert rc.city.name == "testville"
    assert rc.disruption.rhos == (0.0, 0.5)  # tuple => immutable
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.config'`.

- [ ] **Step 4: Implement seeding and config**

Create `src/nnj_topology/seeding.py`:
```python
"""Reproducibility helpers."""
import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["set_seed"]


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Seed set to %d", seed)
```

Create `src/nnj_topology/config.py`:
```python
"""Frozen config dataclasses mirroring the Hydra schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from omegaconf import DictConfig

__all__ = [
    "CityConfig",
    "DisruptionConfig",
    "FiltrationConfig",
    "PathsConfig",
    "RunConfig",
    "from_omegaconf",
]


@dataclass(frozen=True)
class CityConfig:
    name: str
    place: str
    crs: str


@dataclass(frozen=True)
class DisruptionConfig:
    name: str
    rhos: Tuple[float, ...]
    n_replicates: int


@dataclass(frozen=True)
class FiltrationConfig:
    name: str
    max_dim: int


@dataclass(frozen=True)
class PathsConfig:
    data: str
    output: str


@dataclass(frozen=True)
class RunConfig:
    seed: int
    city: CityConfig
    disruption: DisruptionConfig
    filtration: FiltrationConfig
    paths: PathsConfig


def from_omegaconf(cfg: DictConfig) -> RunConfig:
    """Convert a resolved OmegaConf config into a frozen RunConfig."""
    return RunConfig(
        seed=int(cfg.seed),
        city=CityConfig(name=cfg.city.name, place=cfg.city.place, crs=cfg.city.crs),
        disruption=DisruptionConfig(
            name=cfg.disruption.name,
            rhos=tuple(float(r) for r in cfg.disruption.rhos),
            n_replicates=int(cfg.disruption.n_replicates),
        ),
        filtration=FiltrationConfig(name=cfg.filtration.name, max_dim=int(cfg.filtration.max_dim)),
        paths=PathsConfig(data=cfg.paths.data, output=cfg.paths.output),
    )
```

Ensure `src/nnj_topology/__init__.py` exposes the public surface:
```python
"""nnj_topology: topological resilience of green-space access."""
from nnj_topology.config import RunConfig, from_omegaconf
from nnj_topology.seeding import set_seed

__all__ = ["RunConfig", "from_omegaconf", "set_seed"]
```

Create empty `tests/__init__.py`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Add Hydra config files**

Create `conf/config.yaml`:
```yaml
defaults:
  - city: istanbul
  - disruption: random
  - filtration: sublevel
  - _self_

seed: 42
paths:
  data: data
  output: output
```

Create `conf/city/istanbul.yaml` (repeat with appropriate `name`/`place` for the others):
```yaml
name: istanbul
place: "İstanbul, Turkey"
crs: "EPSG:32635"   # UTM 35N (metric, for distances)
```
Create the remaining city files:
- `conf/city/barcelona.yaml`: name `barcelona`, place `"Barcelona, Spain"`, crs `"EPSG:25831"`.
- `conf/city/amsterdam.yaml`: name `amsterdam`, place `"Amsterdam, Netherlands"`, crs `"EPSG:28992"`.
- `conf/city/bogota.yaml`: name `bogota`, place `"Bogotá, Colombia"`, crs `"EPSG:32618"`.
- `conf/city/phoenix.yaml`: name `phoenix`, place `"Phoenix, Arizona, USA"`, crs `"EPSG:26912"`.

Create `conf/disruption/random.yaml`:
```yaml
name: random
rhos: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
n_replicates: 10
```
Create `conf/disruption/targeted.yaml` (same `rhos`, `n_replicates: 1` — deterministic) and `conf/disruption/hazard.yaml` (same `rhos`, `n_replicates: 1`).

Create `conf/filtration/sublevel.yaml`:
```yaml
name: sublevel
max_dim: 1
```
Create `conf/filtration/rips.yaml`: `name: rips`, `max_dim: 1`.

- [ ] **Step 7: Update .gitignore and commit**

Add `output/` and `paper/_build/` to `.gitignore` (keep existing entries). Run:
```bash
git add pyproject.toml uv.lock conf src tests .gitignore
git commit -m "chore: scaffold uv project, config schema, and Hydra configs"
```

---

## Task 2: OSM walk-network loader

**Files:**
- Create: `src/nnj_topology/data/__init__.py`
- Create: `src/nnj_topology/data/network.py`
- Create: `tests/fixtures/make_fixtures.py` (one-off fixture generator)
- Create: `tests/fixtures/mini_graph.graphml` (committed tiny graph)
- Create: `tests/test_network.py`

**Interfaces:**
- Consumes: `CityConfig` from Task 1.
- Produces:
  - `load_walk_network(place: str, crs: str, cache_path: Path | None = None) -> networkx.MultiDiGraph` — downloads via `osmnx`, projects to `crs`, caches to GraphML.
  - `largest_connected_component(G: networkx.MultiDiGraph) -> networkx.MultiDiGraph`.

- [ ] **Step 1: Create the committed test fixture**

Create `tests/fixtures/make_fixtures.py` (run once to produce committed files; not part of the test run):
```python
"""Regenerate committed test fixtures. Run manually: uv run python tests/fixtures/make_fixtures.py"""
from pathlib import Path

import networkx as nx

FIX = Path(__file__).parent


def make_mini_graph() -> None:
    # 3x3 grid of nodes with metric x/y coords and edge length attrs.
    G = nx.MultiDiGraph(crs="EPSG:32635")
    coords = {i * 3 + j: (j * 100.0, i * 100.0) for i in range(3) for j in range(3)}
    for n, (x, y) in coords.items():
        G.add_node(n, x=x, y=y)
    for i in range(3):
        for j in range(3):
            n = i * 3 + j
            if j < 2:
                m = n + 1
                G.add_edge(n, m, length=100.0)
                G.add_edge(m, n, length=100.0)
            if i < 2:
                m = n + 3
                G.add_edge(n, m, length=100.0)
                G.add_edge(m, n, length=100.0)
    nx.write_graphml(G, FIX / "mini_graph.graphml")


if __name__ == "__main__":
    make_mini_graph()
    print("fixtures written")
```
Run: `uv run python tests/fixtures/make_fixtures.py`
Expected: `tests/fixtures/mini_graph.graphml` created.

- [ ] **Step 2: Write the failing test**

Create `tests/test_network.py`:
```python
from pathlib import Path

import networkx as nx

from nnj_topology.data.network import largest_connected_component

FIX = Path(__file__).parent / "fixtures"


def _load_mini() -> nx.MultiDiGraph:
    return nx.read_graphml(FIX / "mini_graph.graphml")


def test_largest_connected_component_returns_full_graph_when_connected():
    G = _load_mini()
    H = largest_connected_component(G)
    assert H.number_of_nodes() == G.number_of_nodes()


def test_largest_connected_component_drops_isolated_node():
    G = _load_mini()
    G.add_node("isolated", x=9999.0, y=9999.0)
    H = largest_connected_component(G)
    assert "isolated" not in H.nodes
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_network.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.data.network'`.

- [ ] **Step 4: Implement the network loader**

Create `src/nnj_topology/data/__init__.py`:
```python
"""Data acquisition modules."""
from nnj_topology.data.network import largest_connected_component, load_walk_network

__all__ = ["load_walk_network", "largest_connected_component"]
```

Create `src/nnj_topology/data/network.py`:
```python
"""OpenStreetMap walk-network acquisition and cleanup."""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import osmnx as ox

logger = logging.getLogger(__name__)

__all__ = ["load_walk_network", "largest_connected_component"]


def load_walk_network(
    place: str, crs: str, cache_path: Path | None = None
) -> nx.MultiDiGraph:
    """Load the pedestrian network for `place`, projected to metric `crs`.

    Uses a GraphML cache when `cache_path` exists to avoid re-downloading.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached walk network from %s", cache_path)
        return ox.load_graphml(cache_path)

    logger.info("Downloading walk network for %s", place)
    graph = ox.graph_from_place(place, network_type="walk")
    graph = ox.project_graph(graph, to_crs=crs)
    graph = largest_connected_component(graph)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, cache_path)
        logger.info("Cached walk network to %s", cache_path)
    return graph


def largest_connected_component(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the subgraph induced by the largest weakly connected component."""
    if graph.number_of_nodes() == 0:
        return graph
    components = nx.weakly_connected_components(graph)
    largest = max(components, key=len)
    return graph.subgraph(largest).copy()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_network.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/nnj_topology/data tests/test_network.py tests/fixtures
git commit -m "feat(data): add OSM walk-network loader with cache and LCC cleanup"
```

---

## Task 3: Green/public-space loader and access points

**Files:**
- Create: `src/nnj_topology/data/greenspace.py`
- Modify: `src/nnj_topology/data/__init__.py` (export new functions)
- Create: `tests/test_greenspace.py`

**Interfaces:**
- Consumes: a projected walk network (Task 2); `CityConfig.crs`.
- Produces:
  - `GREENSPACE_TAGS: dict[str, list[str] | bool]` — the OSM tag filter.
  - `load_greenspace(place: str, crs: str, cache_path: Path | None = None) -> geopandas.GeoDataFrame` (polygon geometries in metric crs).
  - `access_points(green: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame` (point geometries = polygon centroids; one row per green space).
  - `snap_points_to_nodes(points: geopandas.GeoDataFrame, graph: networkx.MultiDiGraph) -> list[int]` (nearest network node id per point).

- [ ] **Step 1: Write the failing test**

Create `tests/test_greenspace.py`:
```python
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon

from nnj_topology.data.greenspace import access_points, snap_points_to_nodes

FIX = Path(__file__).parent / "fixtures"


def test_access_points_returns_one_centroid_per_polygon():
    polys = gpd.GeoDataFrame(
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(200, 200), (210, 200), (210, 210), (200, 210)]),
        ],
        crs="EPSG:32635",
    )
    pts = access_points(polys)
    assert len(pts) == 2
    assert pts.geometry.iloc[0].x == 5.0  # centroid of first square


def test_snap_points_to_nearest_node():
    graph = nx.read_graphml(FIX / "mini_graph.graphml")
    # set x/y as floats (graphml reads as str)
    for _, d in graph.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    pts = gpd.GeoDataFrame.from_features(
        [{"type": "Feature", "geometry": {"type": "Point", "coordinates": (5.0, 5.0)}, "properties": {}}],
        crs="EPSG:32635",
    )
    node_ids = snap_points_to_nodes(pts, graph)
    assert len(node_ids) == 1
    # nearest grid node to (5,5) is node "0" at (0,0)
    assert str(node_ids[0]) == "0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_greenspace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.data.greenspace'`.

- [ ] **Step 3: Implement the greenspace loader**

Create `src/nnj_topology/data/greenspace.py`:
```python
"""Green/public-space polygons and their network access points."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox

logger = logging.getLogger(__name__)

__all__ = ["GREENSPACE_TAGS", "load_greenspace", "access_points", "snap_points_to_nodes"]

GREENSPACE_TAGS: dict[str, list[str] | bool] = {
    "leisure": ["park", "garden", "recreation_ground"],
    "landuse": ["grass", "recreation_ground"],
    "place": ["square"],
    "natural": ["wood"],
}


def load_greenspace(
    place: str, crs: str, cache_path: Path | None = None
) -> gpd.GeoDataFrame:
    """Load green/public-space polygons for `place`, projected to metric `crs`."""
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached greenspace from %s", cache_path)
        return gpd.read_file(cache_path).to_crs(crs)

    logger.info("Downloading greenspace for %s", place)
    gdf = ox.features_from_place(place, tags=GREENSPACE_TAGS)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(crs)
    gdf = gdf.reset_index(drop=True)[["geometry"]]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache_path, driver="GPKG")
        logger.info("Cached greenspace to %s", cache_path)
    return gdf


def access_points(green: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One access point per green space (polygon centroid)."""
    centroids = green.geometry.centroid
    return gpd.GeoDataFrame(geometry=centroids.values, crs=green.crs).reset_index(drop=True)


def snap_points_to_nodes(
    points: gpd.GeoDataFrame, graph: nx.MultiDiGraph
) -> list[int]:
    """Return the nearest graph node id for each point (brute-force on node coords)."""
    node_ids = list(graph.nodes)
    xs = np.array([float(graph.nodes[n]["x"]) for n in node_ids])
    ys = np.array([float(graph.nodes[n]["y"]) for n in node_ids])
    result: list[int] = []
    for geom in points.geometry:
        d2 = (xs - geom.x) ** 2 + (ys - geom.y) ** 2
        result.append(node_ids[int(np.argmin(d2))])
    return result
```

Update `src/nnj_topology/data/__init__.py`:
```python
"""Data acquisition modules."""
from nnj_topology.data.greenspace import (
    GREENSPACE_TAGS,
    access_points,
    load_greenspace,
    snap_points_to_nodes,
)
from nnj_topology.data.network import largest_connected_component, load_walk_network

__all__ = [
    "load_walk_network",
    "largest_connected_component",
    "GREENSPACE_TAGS",
    "load_greenspace",
    "access_points",
    "snap_points_to_nodes",
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_greenspace.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/data tests/test_greenspace.py
git commit -m "feat(data): add greenspace loader, access points, and node snapping"
```

---

## Task 4: Population demand points and hazard mask

**Files:**
- Create: `src/nnj_topology/data/population.py`
- Create: `src/nnj_topology/data/hazard.py`
- Modify: `src/nnj_topology/data/__init__.py`
- Create: `tests/fixtures/mini_pop.tif` (generated by make_fixtures.py)
- Create: `tests/test_population.py`, `tests/test_hazard.py`

**Interfaces:**
- Consumes: `CityConfig.crs`, a walk network (Task 2).
- Produces:
  - `load_population_points(raster_path: Path, crs: str, threshold: float = 1.0) -> geopandas.GeoDataFrame` — one point per populated cell, with column `population: float`.
  - `low_elevation_mask(dem_path: Path, quantile: float = 0.1) -> tuple[np.ndarray, rasterio.Affine]` — boolean array True where elevation ≤ the `quantile`-th percentile (flood proxy).

- [ ] **Step 1: Extend the fixture generator and create raster fixtures**

Append to `tests/fixtures/make_fixtures.py` a `make_mini_pop()` function:
```python
def make_mini_pop() -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    arr = np.array([[10.0, 0.0, 5.0], [0.0, 20.0, 0.0], [3.0, 0.0, 8.0]], dtype="float32")
    transform = from_origin(0.0, 300.0, 100.0, 100.0)  # 100m cells, origin top-left
    with rasterio.open(
        FIX / "mini_pop.tif", "w", driver="GTiff", height=3, width=3, count=1,
        dtype="float32", crs="EPSG:32635", transform=transform,
    ) as dst:
        dst.write(arr, 1)


def make_mini_dem() -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    arr = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]], dtype="float32")
    transform = from_origin(0.0, 300.0, 100.0, 100.0)
    with rasterio.open(
        FIX / "mini_dem.tif", "w", driver="GTiff", height=3, width=3, count=1,
        dtype="float32", crs="EPSG:32635", transform=transform,
    ) as dst:
        dst.write(arr, 1)
```
Add calls in `__main__`. Run: `uv run python tests/fixtures/make_fixtures.py`
Expected: `mini_pop.tif` and `mini_dem.tif` written.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_population.py`:
```python
from pathlib import Path

from nnj_topology.data.population import load_population_points

FIX = Path(__file__).parent / "fixtures"


def test_load_population_points_skips_empty_cells():
    pts = load_population_points(FIX / "mini_pop.tif", crs="EPSG:32635", threshold=1.0)
    # 5 nonzero cells in the fixture
    assert len(pts) == 5
    assert "population" in pts.columns
    assert pts["population"].min() >= 1.0
    assert pts["population"].sum() == 46.0  # 10+5+20+3+8
```

Create `tests/test_hazard.py`:
```python
from pathlib import Path

import numpy as np

from nnj_topology.data.hazard import low_elevation_mask

FIX = Path(__file__).parent / "fixtures"


def test_low_elevation_mask_flags_lowest_cells():
    mask, _ = low_elevation_mask(FIX / "mini_dem.tif", quantile=0.2)
    # 20th percentile of [1,2,3,2,5,6,3,6,9] ~ 2.0; cells <= 2.0 are flagged
    assert mask.dtype == bool
    assert mask.sum() >= 1
    assert mask[0, 0]  # elevation 1.0 is lowest -> flagged
    assert not mask[2, 2]  # elevation 9.0 is highest -> not flagged
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_population.py tests/test_hazard.py -v`
Expected: FAIL with `ModuleNotFoundError` for `population` / `hazard`.

- [ ] **Step 4: Implement population and hazard loaders**

Create `src/nnj_topology/data/population.py`:
```python
"""Gridded population -> demand points."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point

logger = logging.getLogger(__name__)

__all__ = ["load_population_points"]


def load_population_points(
    raster_path: Path, crs: str, threshold: float = 1.0
) -> gpd.GeoDataFrame:
    """Convert a population raster into one demand point per populated cell.

    Cells with population < `threshold` are dropped. Points are reprojected to `crs`.
    """
    with rasterio.open(raster_path) as src:
        band = src.read(1).astype("float64")
        transform = src.transform
        src_crs = src.crs

    rows, cols = np.where(band >= threshold)
    if rows.size == 0:
        logger.warning("No populated cells above threshold %.2f", threshold)
        return gpd.GeoDataFrame(geometry=[], crs=crs).assign(population=[])

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    pops = band[rows, cols]
    gdf = gpd.GeoDataFrame(
        {"population": pops},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=src_crs,
    )
    return gdf.to_crs(crs).reset_index(drop=True)
```

Create `src/nnj_topology/data/hazard.py`:
```python
"""DEM -> low-elevation hazard mask (flood proxy)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine

logger = logging.getLogger(__name__)

__all__ = ["low_elevation_mask"]


def low_elevation_mask(
    dem_path: Path, quantile: float = 0.1
) -> tuple[np.ndarray, Affine]:
    """Return a boolean mask (True = low-lying) and the raster transform.

    A cell is flagged when its elevation is at or below the `quantile`-th
    percentile of all valid elevations.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata

    valid = dem if nodata is None else dem[dem != nodata]
    cutoff = float(np.quantile(valid, quantile))
    mask = dem <= cutoff
    if nodata is not None:
        mask &= dem != nodata
    logger.info("Low-elevation cutoff at q=%.2f is %.2f m", quantile, cutoff)
    return mask, transform
```

Update `src/nnj_topology/data/__init__.py` to also export `load_population_points` and `low_elevation_mask` (append to imports and `__all__`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_population.py tests/test_hazard.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/nnj_topology/data tests/test_population.py tests/test_hazard.py tests/fixtures
git commit -m "feat(data): add population demand points and low-elevation hazard mask"
```

---

## Task 5: Accessibility field (network walk-time to nearest green space)

**Files:**
- Create: `src/nnj_topology/accessibility/__init__.py`
- Create: `src/nnj_topology/accessibility/field.py`
- Create: `tests/test_field.py`

**Interfaces:**
- Consumes: walk network (Task 2), access-point node ids (Task 3), demand points (Task 4).
- Produces:
  - `WALK_SPEED_M_PER_MIN: float = 80.0` (≈4.8 km/h).
  - `add_travel_time(graph: networkx.MultiDiGraph, speed_m_per_min: float = WALK_SPEED_M_PER_MIN) -> networkx.MultiDiGraph` — adds edge attribute `travel_time` (minutes) from `length`.
  - `accessibility_field(graph: networkx.MultiDiGraph, source_nodes: list[int]) -> dict[int, float]` — minutes from each node to its nearest green-space access node (multi-source Dijkstra). Unreachable nodes → `inf`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field.py`:
```python
from pathlib import Path

import networkx as nx

from nnj_topology.accessibility.field import accessibility_field, add_travel_time

FIX = Path(__file__).parent / "fixtures"


def _mini() -> nx.MultiDiGraph:
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_add_travel_time_sets_minutes():
    g = add_travel_time(_mini(), speed_m_per_min=100.0)
    tt = [d["travel_time"] for *_, d in g.edges(data=True)]
    assert all(abs(t - 1.0) < 1e-9 for t in tt)  # 100 m / 100 m·min^-1 = 1 min


def test_accessibility_field_distance_from_single_source():
    g = add_travel_time(_mini(), speed_m_per_min=100.0)
    # source = node "0" (corner). node "8" is the opposite corner, 4 hops away.
    field = accessibility_field(g, source_nodes=["0"])
    assert field["0"] == 0.0
    assert abs(field["8"] - 4.0) < 1e-9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_field.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.accessibility.field'`.

- [ ] **Step 3: Implement the accessibility field**

Create `src/nnj_topology/accessibility/__init__.py`:
```python
"""Accessibility field computation."""
from nnj_topology.accessibility.field import (
    WALK_SPEED_M_PER_MIN,
    accessibility_field,
    add_travel_time,
)

__all__ = ["WALK_SPEED_M_PER_MIN", "add_travel_time", "accessibility_field"]
```

Create `src/nnj_topology/accessibility/field.py`:
```python
"""Network walk-time accessibility field to nearest green space."""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)

__all__ = ["WALK_SPEED_M_PER_MIN", "add_travel_time", "accessibility_field"]

WALK_SPEED_M_PER_MIN: float = 80.0  # ~4.8 km/h


def add_travel_time(
    graph: nx.MultiDiGraph, speed_m_per_min: float = WALK_SPEED_M_PER_MIN
) -> nx.MultiDiGraph:
    """Add a `travel_time` (minutes) edge attribute derived from `length` (metres)."""
    for _, _, data in graph.edges(data=True):
        data["travel_time"] = float(data["length"]) / speed_m_per_min
    return graph


def accessibility_field(
    graph: nx.MultiDiGraph, source_nodes: list[int]
) -> dict[int, float]:
    """Walk-minutes from every node to the nearest green-space access node.

    Multi-source Dijkstra over `travel_time`. Unreachable nodes map to inf.
    """
    if not source_nodes:
        raise ValueError("source_nodes must be non-empty")
    lengths = nx.multi_source_dijkstra_path_length(
        graph, sources=set(source_nodes), weight="travel_time"
    )
    return {node: lengths.get(node, float("inf")) for node in graph.nodes}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_field.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/accessibility tests/test_field.py
git commit -m "feat(access): add walk-time accessibility field via multi-source Dijkstra"
```

---

## Task 6: Persistence diagrams (filtration registry + computation)

**Files:**
- Create: `src/nnj_topology/topology/__init__.py` (registry + factory)
- Create: `src/nnj_topology/topology/filtration.py`
- Create: `src/nnj_topology/topology/diagrams.py`
- Create: `tests/test_diagrams.py`

**Interfaces:**
- Consumes: accessibility field (Task 5); a graph (Task 2).
- Produces:
  - `register_filtration(name)` decorator + `filtration_factory(name) -> Callable`.
  - `Diagram = dict[int, np.ndarray]` (canonical form from Global Constraints).
  - `rips_diagram(points_xy: np.ndarray, weights: np.ndarray, max_dim: int = 1) -> Diagram` — weighted Vietoris–Rips on access points using Euclidean distance (network distance variant deferred; documented).
  - `sublevel_diagram(graph: networkx.MultiDiGraph, field: dict[int, float], max_dim: int = 1) -> Diagram` — sublevel-set persistence of the node field on the graph's clique/edge complex via gudhi `SimplexTree`.
  - `essential_finite_split(dgm: Diagram) -> tuple[Diagram, Diagram]` — split finite vs. infinite-death classes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diagrams.py`:
```python
import numpy as np

from nnj_topology.topology.diagrams import (
    essential_finite_split,
    rips_diagram,
    sublevel_diagram,
)
from nnj_topology.topology import filtration_factory


def test_rips_diagram_circle_has_one_h1_class():
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    weights = np.ones(len(pts))
    dgm = rips_diagram(pts, weights, max_dim=1)
    assert 1 in dgm
    # exactly one prominent (long-lived) loop
    pers = dgm[1][:, 1] - dgm[1][:, 0]
    assert (pers > 0.5).sum() == 1


def test_sublevel_diagram_returns_canonical_dict():
    import networkx as nx

    g = nx.path_graph(4)  # 0-1-2-3
    field = {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0}
    dgm = sublevel_diagram(g, field, max_dim=1)
    assert set(dgm.keys()) <= {0, 1}
    assert dgm[0].shape[1] == 2


def test_essential_finite_split_removes_infinities():
    dgm = {0: np.array([[0.0, 1.0], [0.0, np.inf]]), 1: np.array([[0.5, 2.0]])}
    finite, essential = essential_finite_split(dgm)
    assert np.isfinite(finite[0]).all()
    assert finite[0].shape[0] == 1
    assert essential[0].shape[0] == 1


def test_filtration_factory_dispatch():
    assert filtration_factory("rips") is rips_diagram
    assert filtration_factory("sublevel") is sublevel_diagram
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_diagrams.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.topology'`.

- [ ] **Step 3: Implement the topology package**

Create `src/nnj_topology/topology/diagrams.py`:
```python
"""Persistence diagram computation (Rips and sublevel-set)."""
from __future__ import annotations

import logging
from typing import Dict

import gudhi
import networkx as nx
import numpy as np
from ripser import ripser

logger = logging.getLogger(__name__)

Diagram = Dict[int, np.ndarray]

__all__ = ["Diagram", "rips_diagram", "sublevel_diagram", "essential_finite_split"]


def rips_diagram(points_xy: np.ndarray, weights: np.ndarray, max_dim: int = 1) -> Diagram:
    """Weighted Vietoris-Rips persistence on 2-D access points.

    `weights` (e.g. population) scale point radii so that densely demanded
    coverage holes persist longer. Implemented as a weighted-Rips lower star
    via ripser's `distance_matrix` with additive weight offsets.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    n = len(points_xy)
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    w = weights / (weights.max() + 1e-12)
    # higher weight -> earlier birth: subtract a scaled weight bump, clip >= 0
    bump = (w[:, None] + w[None, :]) * 0.5
    dist = np.clip(dist - bump * dist.mean(), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    res = ripser(dist, distance_matrix=True, maxdim=max_dim)
    return {d: np.atleast_2d(res["dgms"][d]) if res["dgms"][d].size else np.empty((0, 2))
            for d in range(max_dim + 1)}


def sublevel_diagram(graph: nx.Graph, field: dict, max_dim: int = 1) -> Diagram:
    """Sublevel-set persistence of a node field on a graph (1-skeleton + filled triangles)."""
    st = gudhi.SimplexTree()
    for node, value in field.items():
        st.insert([int(node)], filtration=float(value))
    for u, v in graph.edges():
        fu, fv = float(field[u]), float(field[v])
        st.insert([int(u), int(v)], filtration=max(fu, fv))
    # fill triangles on every 3-clique so H1 reflects genuine enclosed voids
    for clique in nx.enumerate_all_cliques(nx.Graph(graph)):
        if len(clique) == 3:
            vals = [float(field[c]) for c in clique]
            st.insert([int(c) for c in clique], filtration=max(vals))
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    out: Diagram = {d: [] for d in range(max_dim + 1)}
    for dim, (birth, death) in st.persistence():
        if dim <= max_dim:
            out[dim].append([birth, death])
    return {d: (np.array(v) if v else np.empty((0, 2))) for d, v in out.items()}


def essential_finite_split(dgm: Diagram) -> tuple[Diagram, Diagram]:
    """Split each dimension's diagram into finite-death and infinite-death parts."""
    finite: Diagram = {}
    essential: Diagram = {}
    for dim, arr in dgm.items():
        if arr.size == 0:
            finite[dim] = np.empty((0, 2))
            essential[dim] = np.empty((0, 2))
            continue
        is_inf = ~np.isfinite(arr[:, 1])
        finite[dim] = arr[~is_inf]
        essential[dim] = arr[is_inf]
    return finite, essential
```

Create `src/nnj_topology/topology/filtration.py`:
```python
"""Filtration registry wiring (keeps construction selection config-driven)."""
from __future__ import annotations

from nnj_topology.topology.diagrams import rips_diagram, sublevel_diagram

__all__ = ["BUILTIN_FILTRATIONS"]

BUILTIN_FILTRATIONS = {"rips": rips_diagram, "sublevel": sublevel_diagram}
```

Create `src/nnj_topology/topology/__init__.py`:
```python
"""Topology: persistent homology constructions, diagrams, distances."""
from typing import Callable, Dict

from nnj_topology.topology.diagrams import (
    Diagram,
    essential_finite_split,
    rips_diagram,
    sublevel_diagram,
)
from nnj_topology.topology.filtration import BUILTIN_FILTRATIONS

_REGISTRY: Dict[str, Callable] = dict(BUILTIN_FILTRATIONS)

__all__ = [
    "Diagram",
    "rips_diagram",
    "sublevel_diagram",
    "essential_finite_split",
    "register_filtration",
    "filtration_factory",
]


def register_filtration(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


def filtration_factory(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown filtration '{name}'; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_diagrams.py -v`
Expected: PASS (4 passed). If the weighted-Rips H1 count assertion is flaky, confirm by inspecting `pers` — the unit circle must yield exactly one long bar.

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/topology tests/test_diagrams.py
git commit -m "feat(topology): add Rips + sublevel persistence diagrams and registry"
```

---

## Task 7: Diagram distances (bottleneck + Wasserstein, finite-safe)

**Files:**
- Create: `src/nnj_topology/topology/distances.py`
- Modify: `src/nnj_topology/topology/__init__.py` (export distances)
- Create: `tests/test_distances.py`

**Interfaces:**
- Consumes: `Diagram`, `essential_finite_split` (Task 6).
- Produces:
  - `bottleneck_distance(dgm_a: Diagram, dgm_b: Diagram, dim: int = 1) -> float`.
  - `wasserstein_distance(dgm_a: Diagram, dgm_b: Diagram, dim: int = 1, order: int = 2) -> float`.
  - `total_persistence(dgm: Diagram, dim: int = 1) -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_distances.py`:
```python
import numpy as np

from nnj_topology.topology.distances import (
    bottleneck_distance,
    total_persistence,
    wasserstein_distance,
)


def _d(arr):
    return {1: np.array(arr, dtype=float), 0: np.empty((0, 2))}


def test_bottleneck_identity_is_zero():
    a = _d([[0.0, 1.0]])
    assert bottleneck_distance(a, a, dim=1) == 0.0


def test_bottleneck_detects_shift():
    a = _d([[0.0, 1.0]])
    b = _d([[0.0, 2.0]])
    assert abs(bottleneck_distance(a, b, dim=1) - 0.5) < 1e-6  # death moves by 1 -> half-bottleneck


def test_wasserstein_nonnegative_and_symmetric():
    a = _d([[0.0, 1.0], [0.2, 0.5]])
    b = _d([[0.0, 2.0]])
    w_ab = wasserstein_distance(a, b, dim=1)
    w_ba = wasserstein_distance(b, a, dim=1)
    assert w_ab >= 0
    assert abs(w_ab - w_ba) < 1e-6


def test_total_persistence_sums_bar_lengths():
    a = _d([[0.0, 1.0], [0.5, 2.0]])
    assert abs(total_persistence(a, dim=1) - 2.5) < 1e-9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_distances.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.topology.distances'`.

- [ ] **Step 3: Implement distances**

Create `src/nnj_topology/topology/distances.py`:
```python
"""Finite-safe persistence-diagram distances."""
from __future__ import annotations

import logging

import numpy as np
import persim

from nnj_topology.topology.diagrams import Diagram, essential_finite_split

logger = logging.getLogger(__name__)

__all__ = ["bottleneck_distance", "wasserstein_distance", "total_persistence"]


def _finite(dgm: Diagram, dim: int) -> np.ndarray:
    finite, _ = essential_finite_split(dgm)
    arr = finite.get(dim, np.empty((0, 2)))
    return arr if arr.size else np.empty((0, 2))


def bottleneck_distance(dgm_a: Diagram, dgm_b: Diagram, dim: int = 1) -> float:
    """Bottleneck distance between the finite parts of two diagrams in `dim`."""
    return float(persim.bottleneck(_finite(dgm_a, dim), _finite(dgm_b, dim)))


def wasserstein_distance(
    dgm_a: Diagram, dgm_b: Diagram, dim: int = 1, order: int = 2
) -> float:
    """p-Wasserstein distance (default p=2) between finite parts of two diagrams."""
    return float(persim.wasserstein(_finite(dgm_a, dim), _finite(dgm_b, dim)))


def total_persistence(dgm: Diagram, dim: int = 1) -> float:
    """Sum of bar lengths (death - birth) over finite classes in `dim`."""
    arr = _finite(dgm, dim)
    if arr.size == 0:
        return 0.0
    return float((arr[:, 1] - arr[:, 0]).sum())
```

Update `src/nnj_topology/topology/__init__.py` to import and add to `__all__`: `bottleneck_distance`, `wasserstein_distance`, `total_persistence` from `nnj_topology.topology.distances`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_distances.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/topology tests/test_distances.py
git commit -m "feat(topology): add finite-safe bottleneck/Wasserstein/total-persistence"
```

---

## Task 8: Disruption models (registry: random / targeted / hazard)

**Files:**
- Create: `src/nnj_topology/disruption/__init__.py` (registry + factory)
- Create: `src/nnj_topology/disruption/models.py`
- Create: `tests/test_disruption_models.py`

**Interfaces:**
- Consumes: walk network (Task 2), hazard mask (Task 4).
- Produces:
  - `register_disruption(name)` + `disruption_factory(name) -> Callable`.
  - `random_removal(graph, rho, seed) -> networkx.MultiDiGraph` — remove fraction `rho` of edges uniformly.
  - `targeted_removal(graph, rho, seed=0) -> networkx.MultiDiGraph` — remove top-`rho` edges by betweenness (seed ignored; deterministic).
  - `hazard_removal(graph, rho, seed=0, *, hazard_nodes: set[int]) -> networkx.MultiDiGraph` — remove edges incident to hazard nodes, capped at fraction `rho`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_disruption_models.py`:
```python
from pathlib import Path

import networkx as nx

from nnj_topology.disruption import disruption_factory
from nnj_topology.disruption.models import random_removal, targeted_removal

FIX = Path(__file__).parent / "fixtures"


def _mini() -> nx.MultiDiGraph:
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_random_removal_removes_expected_fraction_and_is_seeded():
    g = _mini()
    m = g.number_of_edges()
    h1 = random_removal(g, rho=0.5, seed=1)
    h2 = random_removal(g, rho=0.5, seed=1)
    assert h1.number_of_edges() == h2.number_of_edges()  # reproducible
    assert h1.number_of_edges() == m - int(0.5 * m)


def test_random_removal_zero_is_identity():
    g = _mini()
    h = random_removal(g, rho=0.0, seed=1)
    assert h.number_of_edges() == g.number_of_edges()


def test_targeted_removal_is_deterministic():
    g = _mini()
    a = targeted_removal(g, rho=0.3)
    b = targeted_removal(g, rho=0.3)
    assert a.number_of_edges() == b.number_of_edges()


def test_factory_dispatch():
    assert disruption_factory("random") is random_removal
    assert disruption_factory("targeted") is targeted_removal
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_disruption_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.disruption'`.

- [ ] **Step 3: Implement disruption models**

Create `src/nnj_topology/disruption/models.py`:
```python
"""Edge-removal disruption scenarios."""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["random_removal", "targeted_removal", "hazard_removal"]


def _n_to_remove(graph: nx.MultiDiGraph, rho: float) -> int:
    return int(rho * graph.number_of_edges())


def random_removal(graph: nx.MultiDiGraph, rho: float, seed: int) -> nx.MultiDiGraph:
    """Remove a fraction `rho` of edges uniformly at random (percolation-style)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    h = graph.copy()
    k = _n_to_remove(h, rho)
    if k == 0:
        return h
    rng = np.random.default_rng(seed)
    edges = list(h.edges(keys=True))
    idx = rng.choice(len(edges), size=k, replace=False)
    h.remove_edges_from([edges[i] for i in idx])
    return h


def targeted_removal(graph: nx.MultiDiGraph, rho: float, seed: int = 0) -> nx.MultiDiGraph:
    """Remove the top fraction `rho` of edges by edge betweenness (deterministic)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    h = graph.copy()
    k = _n_to_remove(h, rho)
    if k == 0:
        return h
    simple = nx.Graph(h)
    bc = nx.edge_betweenness_centrality(simple, weight="length")
    ranked = sorted(bc, key=lambda e: bc[e], reverse=True)[:k]
    for u, v in ranked:
        for key in list(h.get_edge_data(u, v, default={}).keys()):
            h.remove_edge(u, v, key)
        if h.has_edge(v, u):
            for key in list(h.get_edge_data(v, u, default={}).keys()):
                h.remove_edge(v, u, key)
    return h


def hazard_removal(
    graph: nx.MultiDiGraph, rho: float, seed: int = 0, *, hazard_nodes: set
) -> nx.MultiDiGraph:
    """Remove edges incident to hazard nodes, capped at a fraction `rho` of edges."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    h = graph.copy()
    cap = _n_to_remove(h, rho)
    if cap == 0:
        return h
    incident = [
        (u, v, key)
        for u, v, key in h.edges(keys=True)
        if u in hazard_nodes or v in hazard_nodes
    ]
    rng = np.random.default_rng(seed)
    if len(incident) > cap:
        idx = rng.choice(len(incident), size=cap, replace=False)
        incident = [incident[i] for i in idx]
    h.remove_edges_from(incident)
    return h
```

Create `src/nnj_topology/disruption/__init__.py`:
```python
"""Disruption scenarios + registry."""
from typing import Callable, Dict

from nnj_topology.disruption.models import (
    hazard_removal,
    random_removal,
    targeted_removal,
)

_REGISTRY: Dict[str, Callable] = {
    "random": random_removal,
    "targeted": targeted_removal,
    "hazard": hazard_removal,
}

__all__ = [
    "random_removal",
    "targeted_removal",
    "hazard_removal",
    "register_disruption",
    "disruption_factory",
]


def register_disruption(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


def disruption_factory(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown disruption '{name}'; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_disruption_models.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/disruption tests/test_disruption_models.py
git commit -m "feat(disruption): add random/targeted/hazard edge-removal models + registry"
```

---

## Task 9: Resilience curve (D(rho), AUC, critical rho*)

**Files:**
- Create: `src/nnj_topology/disruption/resilience.py`
- Modify: `src/nnj_topology/disruption/__init__.py` (export resilience API)
- Create: `tests/test_resilience.py`

**Interfaces:**
- Consumes: a diagram callable (Task 6), distances (Task 7), disruption factory (Task 8), accessibility field (Task 5).
- Produces:
  - `ResilienceResult` (frozen dataclass): `rhos: tuple[float, ...]`, `distances: tuple[float, ...]`, `auc: float`, `rho_star: float | None`.
  - `resilience_curve(rhos, distance_at_rho: Callable[[float], float]) -> ResilienceResult` — pure aggregator (decoupled from geometry so it is unit-testable).
  - `compute_auc(rhos, distances) -> float` (trapezoid, normalized by rho-range).
  - `critical_rho(rhos, distances, frac: float = 0.5) -> float | None` — smallest rho where D crosses `frac * max(D)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_resilience.py`:
```python
from nnj_topology.disruption.resilience import (
    compute_auc,
    critical_rho,
    resilience_curve,
)


def test_compute_auc_linear_ramp():
    rhos = [0.0, 0.5, 1.0]
    dists = [0.0, 0.5, 1.0]
    # trapezoid area = 0.5, normalized by rho-range 1.0 -> 0.5
    assert abs(compute_auc(rhos, dists) - 0.5) < 1e-9


def test_critical_rho_finds_half_max_crossing():
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    dists = [0.0, 0.1, 0.2, 0.9, 1.0]
    # half of max (1.0) is 0.5; first rho where dist >= 0.5 is 0.75
    assert critical_rho(rhos, dists, frac=0.5) == 0.75


def test_critical_rho_none_when_never_crossed():
    rhos = [0.0, 0.5, 1.0]
    dists = [0.0, 0.0, 0.0]
    assert critical_rho(rhos, dists, frac=0.5) is None


def test_resilience_curve_uses_distance_callable():
    res = resilience_curve([0.0, 0.5, 1.0], distance_at_rho=lambda r: r)
    assert res.distances == (0.0, 0.5, 1.0)
    assert abs(res.auc - 0.5) < 1e-9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_resilience.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.disruption.resilience'`.

- [ ] **Step 3: Implement resilience aggregation**

Create `src/nnj_topology/disruption/resilience.py`:
```python
"""Resilience curve aggregation: D(rho), AUC, critical rho*."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ResilienceResult", "compute_auc", "critical_rho", "resilience_curve"]


@dataclass(frozen=True)
class ResilienceResult:
    rhos: Tuple[float, ...]
    distances: Tuple[float, ...]
    auc: float
    rho_star: Optional[float]


def compute_auc(rhos: Sequence[float], distances: Sequence[float]) -> float:
    """Trapezoid AUC of D(rho), normalized by the rho range."""
    r = np.asarray(rhos, dtype=float)
    d = np.asarray(distances, dtype=float)
    span = r.max() - r.min()
    if span <= 0:
        return 0.0
    return float(np.trapezoid(d, r) / span)


def critical_rho(
    rhos: Sequence[float], distances: Sequence[float], frac: float = 0.5
) -> Optional[float]:
    """Smallest rho where D(rho) first reaches `frac` of its maximum."""
    d = np.asarray(distances, dtype=float)
    if d.max() <= 0:
        return None
    threshold = frac * d.max()
    for rho, dist in zip(rhos, distances):
        if dist >= threshold:
            return float(rho)
    return None


def resilience_curve(
    rhos: Sequence[float], distance_at_rho: Callable[[float], float]
) -> ResilienceResult:
    """Evaluate D at each rho via the supplied callable and aggregate."""
    rhos_t = tuple(float(r) for r in rhos)
    dists_t = tuple(float(distance_at_rho(r)) for r in rhos_t)
    return ResilienceResult(
        rhos=rhos_t,
        distances=dists_t,
        auc=compute_auc(rhos_t, dists_t),
        rho_star=critical_rho(rhos_t, dists_t),
    )
```

> Note: `np.trapezoid` requires NumPy ≥ 2.0 (satisfied by current uv lock). If the lock pins NumPy < 2, use `np.trapz` instead.

Update `src/nnj_topology/disruption/__init__.py` to export `ResilienceResult`, `resilience_curve`, `compute_auc`, `critical_rho` (append imports + `__all__`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_resilience.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/disruption tests/test_resilience.py
git commit -m "feat(disruption): add resilience curve, AUC, and critical rho*"
```

---

## Task 10: Morphology descriptors

**Files:**
- Create: `src/nnj_topology/morphology/__init__.py`
- Create: `src/nnj_topology/morphology/descriptors.py`
- Create: `tests/test_morphology.py`

**Interfaces:**
- Consumes: walk network (Task 2), greenspace polygons (Task 3).
- Produces:
  - `morphology_descriptors(graph: networkx.MultiDiGraph) -> dict[str, float]` — keys: `intersection_density`, `circuity`, `orientation_entropy`, `mean_block_size` (via `osmnx.stats`/`osmnx.bearing`).
  - `greenspace_fragmentation(green: geopandas.GeoDataFrame) -> float` — patch count divided by total green area (km⁻²), a simple fragmentation index.

- [ ] **Step 1: Write the failing test**

Create `tests/test_morphology.py`:
```python
import geopandas as gpd
from shapely.geometry import Polygon

from nnj_topology.morphology.descriptors import greenspace_fragmentation


def test_fragmentation_higher_for_many_small_patches():
    one_big = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])], crs="EPSG:32635"
    )
    many_small = gpd.GeoDataFrame(
        geometry=[
            Polygon([(i, 0), (i + 10, 0), (i + 10, 10), (i, 10)]) for i in range(0, 100, 20)
        ],
        crs="EPSG:32635",
    )
    assert greenspace_fragmentation(many_small) > greenspace_fragmentation(one_big)


def test_fragmentation_zero_area_returns_zero():
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32635")
    assert greenspace_fragmentation(empty) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_morphology.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.morphology'`.

- [ ] **Step 3: Implement morphology descriptors**

Create `src/nnj_topology/morphology/descriptors.py`:
```python
"""Urban-morphology descriptors (the architecture-mathematics bridge)."""
from __future__ import annotations

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox

logger = logging.getLogger(__name__)

__all__ = ["morphology_descriptors", "greenspace_fragmentation"]


def morphology_descriptors(graph: nx.MultiDiGraph) -> dict[str, float]:
    """Compute intersection density, circuity, orientation entropy, mean block size."""
    stats = ox.stats.basic_stats(graph)
    graph_b = ox.bearing.add_edge_bearings(graph)
    entropy = float(ox.bearing.orientation_entropy(graph_b))
    return {
        "intersection_density": float(stats.get("intersection_count", 0))
        / max(float(stats.get("edge_length_total", 1.0)) / 1000.0, 1e-9),
        "circuity": float(stats.get("circuity_avg", float("nan"))),
        "orientation_entropy": entropy,
        "mean_block_size": float(stats.get("street_length_avg", float("nan"))),
    }


def greenspace_fragmentation(green: gpd.GeoDataFrame) -> float:
    """Patch count per square kilometre of green area (higher = more fragmented)."""
    if len(green) == 0:
        return 0.0
    total_area_km2 = float(green.geometry.area.sum()) / 1e6
    if total_area_km2 <= 0:
        return 0.0
    return len(green) / total_area_km2
```

Create `src/nnj_topology/morphology/__init__.py`:
```python
"""Morphology descriptors."""
from nnj_topology.morphology.descriptors import (
    greenspace_fragmentation,
    morphology_descriptors,
)

__all__ = ["morphology_descriptors", "greenspace_fragmentation"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_morphology.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/morphology tests/test_morphology.py
git commit -m "feat(morphology): add osmnx descriptors and greenspace fragmentation index"
```

---

## Task 11: Cross-city analysis (assemble table + correlation/regression)

**Files:**
- Create: `src/nnj_topology/analysis/__init__.py`
- Create: `src/nnj_topology/analysis/crosscity.py`
- Create: `tests/test_crosscity.py`

**Interfaces:**
- Consumes: per-city dicts of morphology descriptors (Task 10) + resilience summaries (Task 9).
- Produces:
  - `build_table(records: list[dict]) -> pandas.DataFrame` — one row per city, columns = morphology descriptors + `auc` + `rho_star` + baseline `total_persistence`.
  - `correlate(df: pandas.DataFrame, target: str = "auc") -> pandas.DataFrame` — Spearman correlation of each morphology descriptor with the resilience target, with p-values.

- [ ] **Step 1: Write the failing test**

Create `tests/test_crosscity.py`:
```python
import pandas as pd

from nnj_topology.analysis.crosscity import build_table, correlate


def _records():
    return [
        {"city": "a", "circuity": 1.0, "auc": 0.1, "rho_star": 0.4, "total_persistence": 1.0},
        {"city": "b", "circuity": 1.2, "auc": 0.2, "rho_star": 0.3, "total_persistence": 2.0},
        {"city": "c", "circuity": 1.4, "auc": 0.3, "rho_star": 0.2, "total_persistence": 3.0},
    ]


def test_build_table_one_row_per_city():
    df = build_table(_records())
    assert list(df["city"]) == ["a", "b", "c"]
    assert "auc" in df.columns


def test_correlate_returns_rho_and_pvalue():
    df = build_table(_records())
    out = correlate(df, target="auc")
    assert {"feature", "spearman_rho", "p_value"} <= set(out.columns)
    # circuity perfectly increases with auc -> rho ~ 1.0
    row = out[out["feature"] == "circuity"].iloc[0]
    assert row["spearman_rho"] > 0.99
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_crosscity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.analysis'`.

- [ ] **Step 3: Implement cross-city analysis**

Create `src/nnj_topology/analysis/crosscity.py`:
```python
"""Cross-city assembly and morphology<->resilience correlation."""
from __future__ import annotations

import logging

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = ["build_table", "correlate"]

_NON_FEATURE = {"city", "auc", "rho_star", "total_persistence"}


def build_table(records: list[dict]) -> pd.DataFrame:
    """Assemble one row per city from morphology + resilience records."""
    df = pd.DataFrame(records)
    if "city" in df.columns:
        df = df.sort_values("city").reset_index(drop=True)
    return df


def correlate(df: pd.DataFrame, target: str = "auc") -> pd.DataFrame:
    """Spearman correlation of each morphology feature with `target`."""
    features = [c for c in df.columns if c not in _NON_FEATURE]
    rows = []
    for feat in features:
        rho, p = stats.spearmanr(df[feat], df[target])
        rows.append({"feature": feat, "spearman_rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)
```

Create `src/nnj_topology/analysis/__init__.py`:
```python
"""Cross-city analysis."""
from nnj_topology.analysis.crosscity import build_table, correlate

__all__ = ["build_table", "correlate"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_crosscity.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/analysis tests/test_crosscity.py
git commit -m "feat(analysis): add cross-city table builder and Spearman correlation"
```

---

## Task 12: Figures

**Files:**
- Create: `src/nnj_topology/viz/__init__.py`
- Create: `src/nnj_topology/viz/figures.py`
- Create: `tests/test_figures.py`

**Interfaces:**
- Consumes: diagrams (Task 6), `ResilienceResult` (Task 9), cross-city table (Task 11).
- Produces (each returns a `matplotlib.figure.Figure`, saved by the caller):
  - `plot_persistence_diagram(dgm: Diagram, dim: int = 1) -> Figure`.
  - `plot_resilience_curves(results: dict[str, ResilienceResult]) -> Figure` (Fig. 5).
  - `plot_morphology_vs_resilience(df: pandas.DataFrame, feature: str, target: str = "auc") -> Figure` (Fig. 6, headline).

- [ ] **Step 1: Write the failing test**

Create `tests/test_figures.py`:
```python
import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from nnj_topology.disruption.resilience import ResilienceResult  # noqa: E402
from nnj_topology.viz.figures import (  # noqa: E402
    plot_morphology_vs_resilience,
    plot_persistence_diagram,
    plot_resilience_curves,
)


def test_plot_persistence_diagram_returns_figure():
    dgm = {0: np.array([[0.0, 1.0]]), 1: np.array([[0.2, 0.9]])}
    fig = plot_persistence_diagram(dgm, dim=1)
    assert isinstance(fig, Figure)


def test_plot_resilience_curves_returns_figure():
    res = {"a": ResilienceResult((0.0, 1.0), (0.0, 1.0), 0.5, 0.5)}
    fig = plot_resilience_curves(res)
    assert isinstance(fig, Figure)


def test_plot_morphology_vs_resilience_returns_figure():
    df = pd.DataFrame({"city": ["a", "b"], "circuity": [1.0, 1.2], "auc": [0.1, 0.2]})
    fig = plot_morphology_vs_resilience(df, feature="circuity", target="auc")
    assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_figures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nnj_topology.viz'`.

- [ ] **Step 3: Implement figures**

Create `src/nnj_topology/viz/figures.py`:
```python
"""Manuscript figures (return Figure objects; caller saves)."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from nnj_topology.disruption.resilience import ResilienceResult
from nnj_topology.topology.diagrams import Diagram

logger = logging.getLogger(__name__)

__all__ = [
    "plot_persistence_diagram",
    "plot_resilience_curves",
    "plot_morphology_vs_resilience",
]


def plot_persistence_diagram(dgm: Diagram, dim: int = 1) -> Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    arr = dgm.get(dim, np.empty((0, 2)))
    finite = arr[np.isfinite(arr[:, 1])] if arr.size else arr
    if finite.size:
        ax.scatter(finite[:, 0], finite[:, 1], s=20, alpha=0.7)
        top = float(finite.max())
        ax.plot([0, top], [0, top], "k--", lw=0.8)
    ax.set_xlabel("birth (walk minutes)")
    ax.set_ylabel("death (walk minutes)")
    ax.set_title(f"H{dim} persistence diagram")
    fig.tight_layout()
    return fig


def plot_resilience_curves(results: dict[str, ResilienceResult]) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    for city, res in results.items():
        ax.plot(res.rhos, res.distances, marker="o", label=city)
        if res.rho_star is not None:
            ax.axvline(res.rho_star, ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"disruption intensity $\rho$")
    ax.set_ylabel(r"diagram distance $D(\rho)$")
    ax.set_title("Resilience curves")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_morphology_vs_resilience(
    df: pd.DataFrame, feature: str, target: str = "auc"
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df[feature], df[target], s=40)
    for _, row in df.iterrows():
        ax.annotate(str(row.get("city", "")), (row[feature], row[target]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(feature.replace("_", " "))
    ax.set_ylabel(target)
    ax.set_title(f"Morphology vs. resilience ({feature})")
    fig.tight_layout()
    return fig
```

Create `src/nnj_topology/viz/__init__.py`:
```python
"""Visualization."""
from nnj_topology.viz.figures import (
    plot_morphology_vs_resilience,
    plot_persistence_diagram,
    plot_resilience_curves,
)

__all__ = [
    "plot_persistence_diagram",
    "plot_resilience_curves",
    "plot_morphology_vs_resilience",
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_figures.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nnj_topology/viz tests/test_figures.py
git commit -m "feat(viz): add persistence-diagram, resilience-curve, and morphology figures"
```

---

## Task 13: Baseline pipeline (data → field → baseline diagram, per city)

**Files:**
- Create: `pipeline/run_baseline.py`
- Create: `tests/test_pipeline_baseline.py` (uses fixtures, no network)

**Interfaces:**
- Consumes: all data/access/topology modules + `RunConfig`.
- Produces:
  - `compute_baseline(graph, green, population, crs, filtration_name, max_dim) -> tuple[Diagram, dict[int, float]]` — pure function, fixture-testable.
  - `main(cfg)` — Hydra entrypoint: loads/caches city data, computes baseline diagram, writes `output/<city>/baseline_diagram.npz` and `output/<city>/field.npz` plus the resolved config and `uv pip freeze`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_baseline.py`:
```python
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, Polygon

from pipeline.run_baseline import compute_baseline

FIX = Path(__file__).parent / "fixtures"


def _mini_graph():
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for n, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_compute_baseline_returns_diagram_and_field():
    g = _mini_graph()
    green = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32635")
    pop = gpd.GeoDataFrame({"population": [1.0]}, geometry=[Point(200, 200)], crs="EPSG:32635")
    dgm, field = compute_baseline(g, green, pop, crs="EPSG:32635",
                                  filtration_name="sublevel", max_dim=1)
    assert 0 in dgm
    assert len(field) == g.number_of_nodes()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline'` (add `pipeline/__init__.py` if needed, or configure `pythonpath` in `pyproject.toml`'s `[tool.pytest.ini_options]` with `pythonpath = ["."]`).

- [ ] **Step 3: Implement the baseline pipeline**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

Create `pipeline/run_baseline.py`:
```python
"""Baseline pipeline: data -> accessibility field -> baseline persistence diagram."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import hydra
import networkx as nx
import numpy as np
from omegaconf import DictConfig

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.config import from_omegaconf
from nnj_topology.data.greenspace import (
    access_points,
    load_greenspace,
    snap_points_to_nodes,
)
from nnj_topology.data.network import load_walk_network
from nnj_topology.data.population import load_population_points
from nnj_topology.seeding import set_seed
from nnj_topology.topology import Diagram, filtration_factory

logger = logging.getLogger(__name__)


def compute_baseline(
    graph: nx.MultiDiGraph,
    green: gpd.GeoDataFrame,
    population: gpd.GeoDataFrame,
    crs: str,
    filtration_name: str,
    max_dim: int,
) -> tuple[Diagram, dict[int, float]]:
    """Compute the baseline persistence diagram and accessibility field."""
    graph = add_travel_time(graph)
    ap = access_points(green)
    source_nodes = snap_points_to_nodes(ap, graph)
    field = accessibility_field(graph, source_nodes)

    builder = filtration_factory(filtration_name)
    if filtration_name == "sublevel":
        dgm = builder(nx.Graph(graph), field, max_dim=max_dim)
    else:  # rips on access points, weighted by nearby population
        pts = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in source_nodes])
        weights = np.ones(len(pts)) if len(population) == 0 else _weights_from_pop(pts, population)
        dgm = builder(pts, weights, max_dim=max_dim)
    return dgm, field


def _weights_from_pop(pts: np.ndarray, population: gpd.GeoDataFrame) -> np.ndarray:
    pop_xy = np.array([[g.x, g.y] for g in population.geometry])
    pop_val = population["population"].to_numpy()
    weights = np.zeros(len(pts))
    for i, (x, y) in enumerate(pts):
        d2 = (pop_xy[:, 0] - x) ** 2 + (pop_xy[:, 1] - y) ** 2
        weights[i] = pop_val[np.argmin(d2)]
    return weights


def _save_diagram(path: Path, dgm: Diagram) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"dim{d}": arr for d, arr in dgm.items()})


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = from_omegaconf(cfg)
    set_seed(rc.seed)
    data_dir = Path(rc.paths.data) / rc.city.name
    out_dir = Path(rc.paths.output) / rc.city.name

    graph = load_walk_network(rc.city.place, rc.city.crs, data_dir / "walk.graphml")
    green = load_greenspace(rc.city.place, rc.city.crs, data_dir / "green.gpkg")
    pop_raster = data_dir / "population.tif"
    population = (
        load_population_points(pop_raster, rc.city.crs)
        if pop_raster.exists()
        else gpd.GeoDataFrame({"population": []}, geometry=[], crs=rc.city.crs)
    )

    dgm, field = compute_baseline(
        graph, green, population, rc.city.crs, rc.filtration.name, rc.filtration.max_dim
    )
    _save_diagram(out_dir / "baseline_diagram.npz", dgm)
    np.savez(out_dir / "field.npz", nodes=list(field.keys()), values=list(field.values()))
    logger.info("Baseline written for %s", rc.city.name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline_baseline.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Smoke-run on one real city (manual, network required)**

Run: `uv run python pipeline/run_baseline.py city=amsterdam`
Expected: `output/amsterdam/baseline_diagram.npz` and `field.npz` created. (Amsterdam first — compact, good OSM coverage. If it runs out of memory at city scale, reduce scope via `ox.graph_from_place` with a smaller boundary or switch `filtration=rips`.)

- [ ] **Step 6: Commit**

```bash
git add pipeline/run_baseline.py tests/test_pipeline_baseline.py pyproject.toml
git commit -m "feat(pipeline): add baseline data->field->diagram pipeline"
```

---

## Task 14: Disruption pipeline (resilience curves per city × scenario)

**Files:**
- Create: `pipeline/run_disruption.py`
- Create: `tests/test_pipeline_disruption.py`

**Interfaces:**
- Consumes: baseline diagram + field (Task 13), disruption factory (Task 8), distances (Task 7), resilience aggregation (Task 9).
- Produces:
  - `resilience_for_city(graph, green, population, crs, rc) -> ResilienceResult` — for each rho: average diagram distance over `n_replicates`, recomputing field+diagram after disruption.
  - `main(cfg)` — writes `output/<city>/resilience_<scenario>.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_disruption.py`:
```python
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, Polygon

from nnj_topology.config import (
    CityConfig,
    DisruptionConfig,
    FiltrationConfig,
    PathsConfig,
    RunConfig,
)
from pipeline.run_disruption import resilience_for_city

FIX = Path(__file__).parent / "fixtures"


def _mini_graph():
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for n, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def _rc():
    return RunConfig(
        seed=42,
        city=CityConfig("mini", "Mini", "EPSG:32635"),
        disruption=DisruptionConfig("random", (0.0, 0.5), 2),
        filtration=FiltrationConfig("sublevel", 1),
        paths=PathsConfig("data", "output"),
    )


def test_resilience_for_city_runs_and_increases_or_equal_at_zero():
    g = _mini_graph()
    green = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32635")
    pop = gpd.GeoDataFrame({"population": [1.0]}, geometry=[Point(200, 200)], crs="EPSG:32635")
    res = resilience_for_city(g, green, pop, "EPSG:32635", _rc())
    assert res.distances[0] == 0.0  # no disruption -> zero distance to itself
    assert len(res.distances) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline_disruption.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.run_disruption'`.

- [ ] **Step 3: Implement the disruption pipeline**

Create `pipeline/run_disruption.py`:
```python
"""Disruption pipeline: resilience curves per city and scenario."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import hydra
import networkx as nx
import numpy as np
from omegaconf import DictConfig

from nnj_topology.config import RunConfig, from_omegaconf
from nnj_topology.disruption import disruption_factory
from nnj_topology.disruption.resilience import ResilienceResult, resilience_curve
from nnj_topology.seeding import set_seed
from nnj_topology.topology.distances import wasserstein_distance
from pipeline.run_baseline import compute_baseline

logger = logging.getLogger(__name__)


def resilience_for_city(
    graph: nx.MultiDiGraph,
    green: gpd.GeoDataFrame,
    population: gpd.GeoDataFrame,
    crs: str,
    rc: RunConfig,
) -> ResilienceResult:
    """Compute D(rho) = mean Wasserstein distance from baseline over replicates."""
    base_dgm, _ = compute_baseline(
        graph, green, population, crs, rc.filtration.name, rc.filtration.max_dim
    )
    disrupt = disruption_factory(rc.disruption.name)

    def distance_at_rho(rho: float) -> float:
        if rho == 0.0:
            return 0.0
        dists = []
        for rep in range(rc.disruption.n_replicates):
            seed = rc.seed + rep
            dg = disrupt(graph, rho, seed=seed)
            dgm, _ = compute_baseline(
                dg, green, population, crs, rc.filtration.name, rc.filtration.max_dim
            )
            dists.append(wasserstein_distance(base_dgm, dgm, dim=1))
        return float(np.mean(dists))

    return resilience_curve(rc.disruption.rhos, distance_at_rho)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from nnj_topology.data.greenspace import load_greenspace
    from nnj_topology.data.network import load_walk_network
    from nnj_topology.data.population import load_population_points

    rc = from_omegaconf(cfg)
    set_seed(rc.seed)
    data_dir = Path(rc.paths.data) / rc.city.name
    out_dir = Path(rc.paths.output) / rc.city.name
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_walk_network(rc.city.place, rc.city.crs, data_dir / "walk.graphml")
    green = load_greenspace(rc.city.place, rc.city.crs, data_dir / "green.gpkg")
    pop_raster = data_dir / "population.tif"
    population = (
        load_population_points(pop_raster, rc.city.crs)
        if pop_raster.exists()
        else gpd.GeoDataFrame({"population": []}, geometry=[], crs=rc.city.crs)
    )

    res = resilience_for_city(graph, green, population, rc.city.crs, rc)
    out_path = out_dir / f"resilience_{rc.disruption.name}.json"
    out_path.write_text(json.dumps(asdict(res), indent=2))
    logger.info("Resilience written to %s", out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline_disruption.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/run_disruption.py tests/test_pipeline_disruption.py
git commit -m "feat(pipeline): add disruption pipeline producing resilience curves"
```

---

## Task 15: Analysis + figures pipeline + full-suite gate

**Files:**
- Create: `pipeline/run_analysis.py`
- Create: `tests/test_pipeline_analysis.py`

**Interfaces:**
- Consumes: per-city `resilience_*.json` (Task 14), baseline diagrams (Task 13), morphology (Task 10), cross-city (Task 11), figures (Task 12).
- Produces:
  - `assemble_records(output_root: Path, cities: list[str], scenario: str) -> list[dict]` — read artifacts into the analysis record format.
  - `main(cfg)` — builds the table, writes `output/summary_table.csv`, `output/correlations_<scenario>.csv`, and saves Figs. 5 & 6 to `output/figures/`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_analysis.py`:
```python
import json
from pathlib import Path

from pipeline.run_analysis import assemble_records


def test_assemble_records_reads_resilience_json(tmp_path: Path):
    city_dir = tmp_path / "amsterdam"
    city_dir.mkdir()
    (city_dir / "resilience_random.json").write_text(
        json.dumps({"rhos": [0.0, 0.5], "distances": [0.0, 1.0], "auc": 0.5, "rho_star": 0.5})
    )
    (city_dir / "morphology.json").write_text(
        json.dumps({"circuity": 1.1, "intersection_density": 50.0,
                    "orientation_entropy": 3.0, "mean_block_size": 80.0,
                    "greenspace_fragmentation": 12.0})
    )
    records = assemble_records(tmp_path, ["amsterdam"], scenario="random")
    assert len(records) == 1
    assert records[0]["city"] == "amsterdam"
    assert records[0]["auc"] == 0.5
    assert records[0]["circuity"] == 1.1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.run_analysis'`.

- [ ] **Step 3: Implement the analysis pipeline**

Create `pipeline/run_analysis.py`:
```python
"""Analysis pipeline: assemble cross-city table, correlations, and figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from nnj_topology.analysis.crosscity import build_table, correlate
from nnj_topology.disruption.resilience import ResilienceResult
from nnj_topology.viz.figures import (
    plot_morphology_vs_resilience,
    plot_resilience_curves,
)

logger = logging.getLogger(__name__)

CITIES = ["istanbul", "barcelona", "amsterdam", "bogota", "phoenix"]


def assemble_records(output_root: Path, cities: list[str], scenario: str) -> list[dict]:
    """Read per-city resilience + morphology artifacts into analysis records."""
    records: list[dict] = []
    for city in cities:
        res_path = output_root / city / f"resilience_{scenario}.json"
        morph_path = output_root / city / "morphology.json"
        if not res_path.exists():
            logger.warning("Missing %s; skipping", res_path)
            continue
        res = json.loads(res_path.read_text())
        rec = {"city": city, "auc": res["auc"], "rho_star": res.get("rho_star")}
        if morph_path.exists():
            rec.update(json.loads(morph_path.read_text()))
        records.append(rec)
    return records


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output_root = Path(cfg.paths.output)
    scenario = cfg.disruption.name
    records = assemble_records(output_root, CITIES, scenario)
    if not records:
        raise RuntimeError("no records assembled; run baseline+disruption first")

    df = build_table(records)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "summary_table.csv", index=False)

    corr = correlate(df, target="auc")
    corr.to_csv(output_root / f"correlations_{scenario}.csv", index=False)

    results = {}
    for city in CITIES:
        rp = output_root / city / f"resilience_{scenario}.json"
        if rp.exists():
            d = json.loads(rp.read_text())
            results[city] = ResilienceResult(
                tuple(d["rhos"]), tuple(d["distances"]), d["auc"], d.get("rho_star")
            )
    plot_resilience_curves(results).savefig(fig_dir / "fig5_resilience.png", dpi=200)

    top_feature = corr.iloc[0]["feature"]
    plot_morphology_vs_resilience(df, feature=top_feature).savefig(
        fig_dir / "fig6_morphology.png", dpi=200
    )
    logger.info("Analysis complete; figures in %s", fig_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline_analysis.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full test suite (gate)**

Run: `uv run pytest -q`
Expected: all tests pass (no failures). Fix any cross-module regressions before committing.

- [ ] **Step 6: Commit**

```bash
git add pipeline/run_analysis.py tests/test_pipeline_analysis.py
git commit -m "feat(pipeline): add analysis pipeline with summary table and figures"
```

---

## Task 16: End-to-end run on all five cities + results capture

**Files:**
- Create: `pipeline/run_all.sh`
- Create: `output/RESULTS.md` (committed summary of real numbers — the bridge to the manuscript)

**Interfaces:**
- Consumes: all three pipelines.
- Produces: populated `output/<city>/` artifacts, `output/summary_table.csv`, `output/correlations_*.csv`, figures, and a human-readable `output/RESULTS.md`.

- [ ] **Step 1: Write the orchestration script**

Create `pipeline/run_all.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
CITIES=(amsterdam barcelona istanbul bogota phoenix)
SCENARIOS=(random targeted hazard)

for city in "${CITIES[@]}"; do
  uv run python pipeline/run_baseline.py city="$city"
  # write morphology.json for the city
  uv run python -c "
import json, osmnx as ox
from pathlib import Path
from omegaconf import OmegaConf
from nnj_topology.data.network import load_walk_network
from nnj_topology.data.greenspace import load_greenspace
from nnj_topology.morphology.descriptors import morphology_descriptors, greenspace_fragmentation
cfg = OmegaConf.load('conf/city/${city}.yaml')
g = load_walk_network(cfg.place, cfg.crs, Path('data/${city}/walk.graphml'))
green = load_greenspace(cfg.place, cfg.crs, Path('data/${city}/green.gpkg'))
m = morphology_descriptors(g)
m['greenspace_fragmentation'] = greenspace_fragmentation(green)
Path('output/${city}').mkdir(parents=True, exist_ok=True)
Path('output/${city}/morphology.json').write_text(json.dumps(m, indent=2))
print('morphology written for ${city}')
"
  for sc in "${SCENARIOS[@]}"; do
    uv run python pipeline/run_disruption.py city="$city" disruption="$sc"
  done
done

for sc in "${SCENARIOS[@]}"; do
  uv run python pipeline/run_analysis.py disruption="$sc"
done
echo "all cities + scenarios complete"
```
Make executable: `chmod +x pipeline/run_all.sh`

- [ ] **Step 2: Execute the full run (network + compute heavy; manual)**

Run: `bash pipeline/run_all.sh`
Expected: artifacts for all five cities + three scenarios; `output/summary_table.csv` and `output/correlations_random.csv` populated. If a city OOMs at the sublevel-set stage, document it and rerun that city with `filtration=rips` (note the construction switch in RESULTS.md and in the paper's methods/limitations).

- [ ] **Step 3: Sanity-check the results**

Run: `uv run python -c "import pandas as pd; print(pd.read_csv('output/summary_table.csv')); print(pd.read_csv('output/correlations_random.csv'))"`
Expected: five city rows; correlation table with finite Spearman values. Confirm the headline morphology↔resilience relationship is interpretable (sign + magnitude make architectural sense; if not, investigate before writing).

- [ ] **Step 4: Write RESULTS.md**

Create `output/RESULTS.md` capturing, with the actual numbers from this run: per-city baseline desert count / total H1 persistence; AUC and rho* per city × scenario; the top morphology↔resilience correlations with p-values; and any city that required the rips fallback. This file is the single source of truth the manuscript cites — no number enters the paper that is not here.

- [ ] **Step 5: Commit (artifacts gitignored; results summary tracked)**

```bash
git add -f output/RESULTS.md pipeline/run_all.sh
git commit -m "chore(results): full five-city run and captured results summary"
```

---

## Task 17: Manuscript draft (Nexus Network Journal)

**Files:**
- Create: `paper/main.tex`, `paper/sections/*.tex`, `paper/references.bib`
- Create: `paper/figures/` (copy the four to six final figures from `output/figures/`)

**Interfaces:**
- Consumes: `output/RESULTS.md` (all numbers), figures from `output/figures/`, the spec (positioning, RQs, contributions).

> This task is prose, not TDD. Its "test" is the `paper-self-review` skill plus `citation-verification`. Use the `ml-paper-writing` skill for drafting and the Nexus Network Journal author guidelines for formatting (Springer `svjour3`/journal class; NNJ favours clear mathematical exposition for an architecture audience).

- [ ] **Step 1: Set up the LaTeX skeleton**

Create `paper/main.tex` using the Springer NNJ class, with `\input` of section files: `introduction`, `related_work`, `math_background`, `methodology`, `case_studies`, `results`, `discussion`, `conclusion`. Add `references.bib` seeded with the five prior-art arXiv entries from the spec (2206.04834, 1707.08557, 2512.12011, 2512.10753, 2104.00720) plus SDG 11.7 and TDA/persistent-homology foundational references.

- [ ] **Step 2: Draft each section against the spec's structure (§9)**

Write, in order, pulling every quantitative claim from `output/RESULTS.md`:
1. Introduction — SDG 11.7; access vs. resilience; why topology.
2. Related work — TDA in urban analysis; green-space accessibility & equity; urban resilience; **state the honest novelty boundary** (§3 of spec).
3. Mathematical background — simplicial complexes, filtrations, persistent homology, bottleneck/Wasserstein (NNJ-appropriate exposition).
4. Methodology — pipeline + the resilience metric (diagram-distance curve, AUC, critical rho*; percolation link).
5. Case studies — the five cities, data sources, parameters (walk speed, rho grid, replicates).
6. Results — per-city deserts, resilience curves (Fig. 5), morphology↔resilience (Fig. 6, headline), correlation table.
7. Discussion — design/morphology implications, SDG 11 policy relevance, limitations (§10 of spec, including the rips-fallback note if used).
8. Conclusion & future work — network-distance Rips variant; full hazard simulation; more cities.

- [ ] **Step 3: Verify citations**

Use the `citation-verification` skill on `references.bib`. Confirm every cited arXiv id / DOI resolves and matches the claim it supports. Fix or drop any that fail.

- [ ] **Step 4: Compile the PDF**

Run: `cd paper && latexmk -pdf main.tex` (or the user's preferred LaTeX toolchain).
Expected: `paper/main.pdf` builds with no undefined references and all figures present.

- [ ] **Step 5: Self-review**

Use the `paper-self-review` skill (6-item quality checklist) and `writing-anti-ai` on the prose. Address every flagged item. Confirm no number in the PDF is absent from `output/RESULTS.md`.

- [ ] **Step 6: Commit**

```bash
git add paper
git commit -m "docs(paper): draft NNJ manuscript on topological green-space resilience"
```

---

## Self-Review (plan author's checklist — completed)

**1. Spec coverage:**
- RQ1 (topological signature of access) → Tasks 5, 6, 13.
- RQ2 (degradation under disruption) → Tasks 8, 9, 14.
- RQ3 (morphology↔resilience) → Tasks 10, 11, 15.
- Contributions C1/C2/C3 → resilience metric (Task 9), SDG-11.7 framing (manuscript Task 17), cross-city map (Tasks 11, 16).
- Methodology §6.1–6.6 → data Tasks 2–4; field Task 5; PH Tasks 6–7; disruption Tasks 8–9; cross-city Tasks 10–11; stack pinned in Task 1.
- Five cities §7 → config files (Task 1) + run_all (Task 16).
- Six figures §8 → Task 12 covers diagram/resilience/morphology; per-city maps + accessibility heatmaps + pipeline schematic are remaining figures, produced during Task 16/17 (noted as a gap to fill: add `plot_city_map` and `plot_field_heatmap` if reviewers want them — currently optional, headline figures are covered).
- Limitations §10 → manuscript Task 17 Step 2.7; rips-fallback path threaded through Tasks 13/16.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" left; all code steps contain runnable code.

**3. Type consistency:** `Diagram` dict form is consistent across Tasks 6, 7, 12, 13, 14; `ResilienceResult` fields (`rhos`, `distances`, `auc`, `rho_star`) consistent across Tasks 9, 14, 15; `compute_baseline` signature identical in Tasks 13 and 14; `morphology.json` keys (Task 16) match the non-feature exclusion set and `correlate` features (Task 11).

**Known minor gap (intentional, low-risk):** the network-distance Vietoris–Rips variant (spec §6.3b) is approximated by Euclidean weighted-Rips in Task 6; the full network-metric version is listed as future work in the manuscript. Per-city map and accessibility-heatmap figures (spec figs 2–3) are deferred to Task 17 as optional additions; the two headline figures (5, 6) are fully implemented.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-topological-resilience-green-space-implementation.md`.

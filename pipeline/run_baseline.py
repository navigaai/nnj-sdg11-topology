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

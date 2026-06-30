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

"""District analysis pipeline: per-district records -> regression -> figures."""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Callable

import geopandas as gpd
import h3
import hydra
import networkx as nx
from omegaconf import DictConfig
from shapely.geometry import Polygon

from nnj_topology.analysis.crosscity import city_typology
from nnj_topology.analysis.regression import (
    build_district_frame,
    fixed_effects_regression,
    tidy_coefficients,
)
from nnj_topology.disruption import disruption_factory
from nnj_topology.disruption.models import betweenness_ranking, targeted_removal
from nnj_topology.disruption.resilience import ResilienceResult
from nnj_topology.districts.tiling import assign_nodes_to_hexes, district_resilience, local_diagram
from nnj_topology.morphology.descriptors import (
    greenspace_fragmentation,
    morphology_descriptors,
)
from nnj_topology.topology.distances import total_persistence
from nnj_topology.viz.figures import (
    plot_morphology_vs_resilience,
    plot_resilience_curves,
)

logger = logging.getLogger(__name__)

CITIES = ["istanbul", "barcelona", "amsterdam", "bogota", "phoenix"]


def _hex_polygon(cell: str, crs: str):
    """Return the H3 hex cell boundary as a shapely geometry in the given CRS.

    H3 v4 ``cell_to_boundary`` returns a list of (lat, lng) pairs.  We convert
    to shapely (x=lng, y=lat) order, wrap in a GeoSeries in EPSG:4326, then
    reproject to the local metric ``crs``.
    """
    ring = h3.cell_to_boundary(cell)
    poly = Polygon([(lng, lat) for lat, lng in ring])
    return gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(crs).iloc[0]


def district_morphology(
    graph: nx.MultiDiGraph, green: gpd.GeoDataFrame, crs: str, hex_nodes: dict
) -> dict:
    """Per-hex morphology descriptors (osmnx stats on the hex subgraph).

    Greenspace fragmentation is computed on the green space CLIPPED to each
    individual hex cell so the metric varies between districts rather than being
    a per-city constant that is collinear with city fixed effects.
    """
    out: dict = {}
    for cell, nodes in hex_nodes.items():
        sub = graph.subgraph(nodes).copy()
        if sub.number_of_edges() == 0:
            continue
        try:
            desc = morphology_descriptors(sub)
        except Exception as exc:  # noqa: BLE001 - osmnx stats fail on tiny subgraphs
            logger.debug("morphology failed for hex %s: %s", cell, exc)
            continue
        # Clip green space to this hex so fragmentation varies per district.
        hexpoly = _hex_polygon(cell, crs)
        candidates = green[green.intersects(hexpoly)]
        if len(candidates) == 0:
            local_green = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=green.crs), crs=green.crs
            )
        else:
            clipped = candidates.geometry.intersection(hexpoly)
            local_green = gpd.GeoDataFrame(
                geometry=clipped[~clipped.is_empty].reset_index(drop=True),
                crs=green.crs,
            )
        desc["greenspace_fragmentation"] = greenspace_fragmentation(local_green)
        out[cell] = desc
    return out


def compute_district_records(
    graph: nx.MultiDiGraph,
    green: gpd.GeoDataFrame,
    field_fn: Callable[[nx.MultiDiGraph], dict],
    disrupt: Callable[..., nx.MultiDiGraph],
    crs: str,
    rc,
    min_nodes: int = 10,
) -> list[dict]:
    """Build one record per qualifying district (hex)."""
    # Precompute betweenness ranking once for targeted scenario so the expensive
    # O(V*E) computation is not repeated across every (rho, replicate) call.
    disrupt_name = getattr(rc.disruption, "name", None)
    if disrupt_name == "targeted":
        _ranking = betweenness_ranking(graph)
        disrupt = functools.partial(targeted_removal, ranking=_ranking)
    elif disrupt_name == "hazard":
        from nnj_topology.data.hazard import hazard_nodes_from_dem
        from nnj_topology.disruption.models import hazard_removal

        dem_path = Path(rc.paths.data) / rc.city.name / "dem.tif"
        _hazard_nodes = hazard_nodes_from_dem(graph, dem_path, crs)
        disrupt = functools.partial(hazard_removal, hazard_nodes=_hazard_nodes)

    hex_nodes = assign_nodes_to_hexes(graph, crs, rc.h3_res)
    res_by_hex = district_resilience(
        graph, field_fn, disrupt,
        rhos=list(rc.disruption.rhos), n_replicates=rc.disruption.n_replicates,
        seed=rc.seed, hex_nodes=hex_nodes, max_dim=rc.filtration.max_dim,
        min_nodes=min_nodes, dim=rc.homology_dim,
        min_persistence=rc.persistence_threshold,
    )
    morph_by_hex = district_morphology(graph, green, crs, hex_nodes)
    base_field = field_fn(graph)
    simple = nx.Graph(graph)

    records: list[dict] = []
    for cell, res in res_by_hex.items():
        if cell not in morph_by_hex:
            continue
        base_dgm = local_diagram(simple, base_field, hex_nodes[cell], rc.filtration.max_dim)
        rec = {"city": rc.city.name, "hex": cell}
        rec.update(morph_by_hex[cell])
        rec["auc"] = res.auc
        rec["rho_star"] = res.rho_star if res.rho_star is not None else float("nan")
        rec["total_persistence"] = total_persistence(base_dgm, dim=rc.homology_dim)
        records.append(rec)
    return records


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from nnj_topology.accessibility.field import accessibility_field, add_travel_time
    from nnj_topology.config import (
        CityConfig,
        RunConfig,
        from_omegaconf,
    )
    from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
    from nnj_topology.data.greenspace import (
        access_points,
        load_greenspace,
        snap_points_to_nodes,
    )
    from nnj_topology.data.network import load_walk_network

    rc = from_omegaconf(cfg)
    output_root = Path(rc.paths.output)
    scenario = rc.disruption.name
    disrupt = disruption_factory(scenario)

    cities = list(cfg.get("cities", CITIES))  # override with cities=[amsterdam] to run a subset
    all_records: list[dict] = []
    curves: dict[str, ResilienceResult] = {}
    for city in cities:
        city_cfg = hydra.compose(config_name="config", overrides=[f"city={city}"]).city
        data_dir = Path(rc.paths.data) / city
        if scenario == "hazard" and not (data_dir / "dem.tif").exists():
            logger.warning(
                "skipping hazard for %s: no DEM at %s/dem.tif", city, data_dir
            )
            continue
        graph = load_walk_network(city_cfg.place, city_cfg.crs, data_dir / "walk.graphml")
        boundary = load_urban_boundary(
            Path(rc.paths.data) / "ghsl" / "ghs_ucdb.gpkg", city_cfg.name, city_cfg.crs
        )
        graph = clip_graph_to_boundary(graph, boundary, city_cfg.crs)
        graph = add_travel_time(graph)
        green = load_greenspace(city_cfg.place, city_cfg.crs, data_dir / "green.gpkg")

        def field_fn(g, _green=green):
            ap = access_points(_green)
            return accessibility_field(g, snap_points_to_nodes(ap, g))

        # h3_res is a real field on RunConfig (added Task 1); construct directly,
        # no object.__setattr__ hack needed.
        rc_city = RunConfig(
            seed=rc.seed,
            h3_res=rc.h3_res,
            homology_dim=rc.homology_dim,
            persistence_threshold=rc.persistence_threshold,
            city=CityConfig(city_cfg.name, city_cfg.place, city_cfg.crs),
            disruption=rc.disruption,
            filtration=rc.filtration,
            paths=rc.paths,
        )
        all_records.extend(
            compute_district_records(graph, green, field_fn, disrupt, city_cfg.crs, rc_city)
        )

        rp = output_root / city / f"resilience_{scenario}.json"
        if rp.exists():
            d = json.loads(rp.read_text())
            curves[city] = ResilienceResult(
                tuple(d["rhos"]), tuple(d["distances"]), d["auc"], d.get("rho_star")
            )

    if not all_records:
        raise RuntimeError("no district records; run baseline first and check boundaries")

    df = build_district_frame(all_records)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "district_table.csv", index=False)

    features = [c for c in df.columns if c not in {"city", "hex", "auc", "rho_star", "total_persistence"}]
    result = fixed_effects_regression(df, target="auc", features=features)
    tidy = tidy_coefficients(result)
    tidy.to_csv(output_root / f"regression_{scenario}.csv", index=False)
    city_typology(df).to_csv(output_root / "city_typology.csv", index=False)

    if curves:
        plot_resilience_curves(curves).savefig(fig_dir / "fig5_resilience.png", dpi=200)
    top_feature = tidy.sort_values("p_value").iloc[0]["term"]
    plot_morphology_vs_resilience(df, feature=top_feature).savefig(
        fig_dir / "fig6_morphology.png", dpi=200
    )
    logger.info("District analysis complete (%d districts); figures in %s", len(df), fig_dir)


if __name__ == "__main__":
    main()

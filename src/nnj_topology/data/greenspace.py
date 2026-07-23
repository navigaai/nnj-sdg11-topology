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
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(crs)
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

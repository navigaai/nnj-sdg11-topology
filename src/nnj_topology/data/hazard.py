"""DEM -> low-elevation hazard mask (flood proxy) and hazard-node extraction."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio import Affine
from shapely.geometry import Point

from nnj_topology.data.greenspace import snap_points_to_nodes

logger = logging.getLogger(__name__)

__all__ = ["low_elevation_mask", "hazard_nodes_from_dem"]


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


def hazard_nodes_from_dem(
    graph: nx.MultiDiGraph, dem_path: Path, crs: str, quantile: float = 0.1
) -> set:
    """Return the set of graph nodes lying in low-elevation (flood-prone) cells.

    Flags the ``quantile``-th percentile lowest DEM cells, reprojects their
    cell centres from the DEM CRS into the graph's metric ``crs``, and snaps
    each to its nearest network node. Returns an empty set if no cells qualify.
    """
    mask, transform = low_elevation_mask(dem_path, quantile)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs

    rows, cols = np.where(mask)
    if rows.size == 0:
        logger.warning("No low-elevation cells found in %s", dem_path)
        return set()

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=dem_crs
    ).to_crs(crs)
    nodes = set(snap_points_to_nodes(points, graph))
    logger.info("Derived %d hazard nodes from %s", len(nodes), dem_path)
    return nodes

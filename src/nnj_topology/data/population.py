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

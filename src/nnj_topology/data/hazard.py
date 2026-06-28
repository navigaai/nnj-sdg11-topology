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

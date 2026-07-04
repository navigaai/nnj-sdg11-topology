"""Fetch a DEM for a city from AWS Terrain Tiles (public, no auth) into
data/<city>/dem.tif, covering the city's GHS-UCDB urban-centre boundary.

Usage: uv run python scripts/fetch_dem.py <city> [zoom]
"""
from __future__ import annotations

import math
import sys
import tempfile
import urllib.request
from pathlib import Path

import geopandas as gpd
import rasterio
from omegaconf import OmegaConf
from rasterio.merge import merge

from nnj_topology.data.boundary import load_urban_boundary

TILE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{z}/{x}/{y}.tif"


def _deg2tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def fetch_dem(city: str, zoom: int = 10) -> Path:
    cfg = OmegaConf.load(f"conf/city/{city}.yaml")
    boundary = load_urban_boundary(
        Path("data/ghsl/ghs_ucdb.gpkg"), cfg.name, cfg.crs
    )
    # bounding box in lat/lon
    ll = gpd.GeoSeries([boundary], crs=cfg.crs).to_crs("EPSG:4326").total_bounds
    minlon, minlat, maxlon, maxlat = ll
    x0, y1 = _deg2tile(minlat, minlon, zoom)  # note: y grows southward
    x1, y0 = _deg2tile(maxlat, maxlon, zoom)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    print(f"{city}: {len(xs)}x{len(ys)} tiles at z{zoom}")

    with tempfile.TemporaryDirectory() as tmp:
        srcs = []
        for x in xs:
            for y in ys:
                dst = Path(tmp) / f"{x}_{y}.tif"
                url = TILE_URL.format(z=zoom, x=x, y=y)
                try:
                    urllib.request.urlretrieve(url, dst)
                    srcs.append(rasterio.open(dst))
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip tile {x}/{y}: {exc}")
        if not srcs:
            raise RuntimeError("no DEM tiles fetched")
        mosaic, transform = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(
            {"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform}
        )
        out = Path("data") / city / "dem.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(mosaic)
        for s in srcs:
            s.close()
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


if __name__ == "__main__":
    city = sys.argv[1]
    zoom = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    fetch_dem(city, zoom)

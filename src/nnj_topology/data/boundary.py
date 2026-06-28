"""GHS-UCDB urban-centre boundary loading and graph clipping."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

__all__ = ["load_urban_boundary", "clip_graph_to_boundary"]


def load_urban_boundary(
    ucdb_path: Path, city_name_match: str, crs: str
) -> BaseGeometry:
    """Return the GHS-UCDB urban-centre polygon matching `city_name_match`.

    Matching is case-insensitive substring on the UCDB name column (`UC_NM_MN`).
    """
    gdf = gpd.read_file(ucdb_path)
    name_col = "UC_NM_MN" if "UC_NM_MN" in gdf.columns else gdf.columns[0]
    hit = gdf[gdf[name_col].astype(str).str.contains(city_name_match, case=False, na=False)]
    if hit.empty:
        raise ValueError(f"no UCDB urban centre matching '{city_name_match}'")
    return hit.to_crs(crs).union_all()


def clip_graph_to_boundary(
    graph: nx.MultiDiGraph, boundary: BaseGeometry, crs: str
) -> nx.MultiDiGraph:
    """Keep only nodes whose (x, y) fall within `boundary`."""
    inside = [
        n
        for n, d in graph.nodes(data=True)
        if boundary.covers(Point(float(d["x"]), float(d["y"])))
    ]
    return graph.subgraph(inside).copy()

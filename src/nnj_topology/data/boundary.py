"""GHS-UCDB urban-centre boundary loading and graph clipping."""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

__all__ = ["load_urban_boundary", "clip_graph_to_boundary"]


def _fold(s: str) -> str:
    """Fold a string to lowercase ASCII by stripping diacritic combining marks."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    ).casefold()


def load_urban_boundary(
    ucdb_path: Path, city_name_match: str, crs: str
) -> BaseGeometry:
    """Return the GHS-UCDB urban-centre polygon matching `city_name_match`.

    Matching is case-insensitive substring on the UCDB name column (`UC_NM_MN`),
    with diacritics folded so that e.g. "bogota" matches "Bogotá". When several
    urban centres share the name (e.g. Barcelona ES and Barcelona VE), the
    single largest-area centre is selected — the intended major city — rather
    than unioning distant homonyms into one geometry.
    """
    gdf = gpd.read_file(ucdb_path)
    name_col = "UC_NM_MN" if "UC_NM_MN" in gdf.columns else gdf.columns[0]
    needle = _fold(city_name_match)
    hit = gdf[gdf[name_col].map(_fold).str.contains(needle, na=False)]
    if hit.empty:
        raise ValueError(f"no UCDB urban centre matching '{city_name_match}'")
    if len(hit) > 1:
        # Rank by TRUE area in an equal-area CRS (World Mollweide). Ranking in
        # the target UTM `crs` wildly distorts a distant homonym (e.g. Barcelona
        # VE projected into Spain's UTM 31N looks huge), picking the wrong city.
        areas = hit.to_crs("ESRI:54009").geometry.area
        hit = hit.loc[[areas.idxmax()]]
        ctr_col = next((c for c in ("CTR_MN_NM", "CTR_MN_ISO") if c in hit.columns), None)
        where = f" ({hit.iloc[0][ctr_col]})" if ctr_col else ""
        logger.info(
            "Multiple UCDB matches for '%s'; picked largest by equal-area%s",
            city_name_match,
            where,
        )
    return hit.to_crs(crs).geometry.iloc[0]


def clip_graph_to_boundary(
    graph: nx.MultiDiGraph, boundary: BaseGeometry, crs: str
) -> nx.MultiDiGraph:
    """Keep only nodes whose (x, y) fall within `boundary`."""
    inside = [
        n
        for n, d in graph.nodes(data=True)
        if boundary.covers(Point(float(d["x"]), float(d["y"])))
    ]
    if not inside:
        raise ValueError(
            "boundary does not overlap the graph (0 nodes inside) — check that "
            "the UCDB match and the network are the same city / CRS"
        )
    return graph.subgraph(inside).copy()

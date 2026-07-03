from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon

from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary

FIX = Path(__file__).parent / "fixtures"


def _mini():
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for _, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    return g


def test_clip_graph_keeps_only_inside_nodes():
    g = _mini()  # 3x3 grid spanning (0,0)-(200,200)
    boundary = Polygon([(-1, -1), (110, -1), (110, 110), (-1, 110)])  # covers nodes 0,1,3,4
    h = clip_graph_to_boundary(g, boundary, crs="EPSG:32635")
    assert set(map(str, h.nodes)) == {"0", "1", "3", "4"}


def test_load_urban_boundary_matches_accented_name(tmp_path):
    """load_urban_boundary("bogota") must match a row whose name is "Bogotá"."""
    bogota_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    barcelona_poly = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    gdf = gpd.GeoDataFrame(
        {"UC_NM_MN": ["Bogotá", "Barcelona"]},
        geometry=[bogota_poly, barcelona_poly],
        crs="EPSG:4326",
    )
    gpkg = tmp_path / "test_ucdb.gpkg"
    gdf.to_file(gpkg, driver="GPKG")

    result = load_urban_boundary(gpkg, "bogota", "EPSG:4326")

    # Must return a geometry (not raise ValueError)
    assert result is not None
    # Must be the Bogotá polygon (not Barcelona): check centroid is inside bogota_poly
    bogota_centroid = bogota_poly.centroid
    assert result.covers(bogota_centroid) or result.contains(bogota_centroid)
    # Must NOT contain the Barcelona centroid
    barcelona_centroid = barcelona_poly.centroid
    assert not result.contains(barcelona_centroid)


def test_load_urban_boundary_disambiguates_homonyms_by_area(tmp_path):
    """Two 'Barcelona' centres (ES big, VE small) must not be unioned; pick the largest."""
    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # "Spain" — large
    small = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])  # "Venezuela" — tiny, far
    gdf = gpd.GeoDataFrame(
        {"UC_NM_MN": ["Barcelona", "Barcelona"], "CTR_MN_NM": ["Spain", "Venezuela"]},
        geometry=[big, small],
        crs="EPSG:4326",
    )
    gpkg = tmp_path / "homonyms.gpkg"
    gdf.to_file(gpkg, driver="GPKG")

    result = load_urban_boundary(gpkg, "barcelona", "EPSG:4326")
    # Picked the large one; the distant small centre is excluded (no transcontinental union).
    assert result.covers(big.centroid)
    assert not result.covers(small.centroid)

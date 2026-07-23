from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon

from nnj_topology.data.greenspace import access_points, snap_points_to_nodes

FIX = Path(__file__).parent / "fixtures"


def test_access_points_returns_one_centroid_per_polygon():
    polys = gpd.GeoDataFrame(
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(200, 200), (210, 200), (210, 210), (200, 210)]),
        ],
        crs="EPSG:32635",
    )
    pts = access_points(polys)
    assert len(pts) == 2
    assert pts.geometry.iloc[0].x == 5.0  # centroid of first square


def test_snap_points_to_nearest_node():
    graph = nx.read_graphml(FIX / "mini_graph.graphml")
    # set x/y as floats (graphml reads as str)
    for _, d in graph.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    pts = gpd.GeoDataFrame.from_features(
        [{"type": "Feature", "geometry": {"type": "Point", "coordinates": (5.0, 5.0)}, "properties": {}}],
        crs="EPSG:32635",
    )
    node_ids = snap_points_to_nodes(pts, graph)
    assert len(node_ids) == 1
    # nearest grid node to (5,5) is node "0" at (0,0)
    assert str(node_ids[0]) == "0"

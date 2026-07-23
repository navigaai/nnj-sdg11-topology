from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, Polygon

from pipeline.run_baseline import compute_baseline

FIX = Path(__file__).parent / "fixtures"


def _mini_graph():
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for n, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_compute_baseline_returns_diagram_and_field():
    g = _mini_graph()
    green = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32635")
    pop = gpd.GeoDataFrame({"population": [1.0]}, geometry=[Point(200, 200)], crs="EPSG:32635")
    dgm, field = compute_baseline(g, green, pop, crs="EPSG:32635",
                                  filtration_name="sublevel", max_dim=1)
    assert 0 in dgm
    assert len(field) == g.number_of_nodes()

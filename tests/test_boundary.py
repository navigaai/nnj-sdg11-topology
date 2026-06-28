from pathlib import Path

import networkx as nx
from shapely.geometry import Polygon

from nnj_topology.data.boundary import clip_graph_to_boundary

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

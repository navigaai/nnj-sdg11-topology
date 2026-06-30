from pathlib import Path

import networkx as nx

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.districts.tiling import assign_nodes_to_hexes, local_diagram

FIX = Path(__file__).parent / "fixtures"


def _mini():
    g = nx.MultiDiGraph(nx.read_graphml(FIX / "mini_graph.graphml"))
    for _, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_assign_nodes_to_hexes_covers_all_nodes():
    g = _mini()
    mapping = assign_nodes_to_hexes(g, crs="EPSG:32635", h3_res=8)
    assigned = sum(len(v) for v in mapping.values())
    assert assigned == g.number_of_nodes()


def test_local_diagram_on_subgraph_returns_canonical_dict():
    g = add_travel_time(_mini(), speed_m_per_min=100.0)
    field = accessibility_field(g, source_nodes=["0"])
    dgm = local_diagram(nx.Graph(g), field, node_ids=["0", "1", "3", "4"], max_dim=1)
    assert 0 in dgm
    assert dgm[0].shape[1] == 2

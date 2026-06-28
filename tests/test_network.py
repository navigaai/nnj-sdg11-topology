from pathlib import Path

import networkx as nx

from nnj_topology.data.network import largest_connected_component

FIX = Path(__file__).parent / "fixtures"


def _load_mini() -> nx.MultiDiGraph:
    return nx.read_graphml(FIX / "mini_graph.graphml")


def test_largest_connected_component_returns_full_graph_when_connected():
    G = _load_mini()
    H = largest_connected_component(G)
    assert H.number_of_nodes() == G.number_of_nodes()


def test_largest_connected_component_drops_isolated_node():
    G = _load_mini()
    G.add_node("isolated", x=9999.0, y=9999.0)
    H = largest_connected_component(G)
    assert "isolated" not in H.nodes

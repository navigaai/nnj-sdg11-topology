from pathlib import Path

import networkx as nx

from nnj_topology.accessibility.field import accessibility_field, add_travel_time

FIX = Path(__file__).parent / "fixtures"


def _mini() -> nx.MultiDiGraph:
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_add_travel_time_sets_minutes():
    g = add_travel_time(_mini(), speed_m_per_min=100.0)
    tt = [d["travel_time"] for *_, d in g.edges(data=True)]
    assert all(abs(t - 1.0) < 1e-9 for t in tt)  # 100 m / 100 m·min^-1 = 1 min


def test_accessibility_field_distance_from_single_source():
    g = add_travel_time(_mini(), speed_m_per_min=100.0)
    # source = node "0" (corner). node "8" is the opposite corner, 4 hops away.
    field = accessibility_field(g, source_nodes=["0"])
    assert field["0"] == 0.0
    assert abs(field["8"] - 4.0) < 1e-9

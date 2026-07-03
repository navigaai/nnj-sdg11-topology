from pathlib import Path

import networkx as nx

from nnj_topology.data.hazard import hazard_nodes_from_dem, low_elevation_mask

FIX = Path(__file__).parent / "fixtures"


def test_low_elevation_mask_flags_lowest_cells():
    mask, _ = low_elevation_mask(FIX / "mini_dem.tif", quantile=0.2)
    # 20th percentile of [1,2,3,2,5,6,3,6,9] ~ 2.0; cells <= 2.0 are flagged
    assert mask.dtype == bool
    assert mask.sum() >= 1
    assert mask[0, 0]  # elevation 1.0 is lowest -> flagged
    assert not mask[2, 2]  # elevation 9.0 is highest -> not flagged


def test_hazard_nodes_from_dem_returns_graph_nodes():
    g = nx.MultiDiGraph(nx.read_graphml(FIX / "mini_graph.graphml"))
    for _, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    nodes = hazard_nodes_from_dem(g, FIX / "mini_dem.tif", "EPSG:32635", quantile=0.4)
    assert isinstance(nodes, set)
    assert len(nodes) >= 1
    assert set(nodes) <= set(g.nodes)  # only real graph nodes returned

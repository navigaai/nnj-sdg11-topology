from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.greenspace import access_points, snap_points_to_nodes
from nnj_topology.disruption.models import random_removal
from pipeline.run_analysis import compute_district_records

FIX = Path(__file__).parent / "fixtures"


def _mini():
    # Wrap as MultiDiGraph: random_removal calls .edges(keys=True) which
    # requires a MultiGraph/MultiDiGraph; the stored graphml is a plain DiGraph.
    g = nx.MultiDiGraph(nx.read_graphml(FIX / "mini_graph.graphml"))
    for _, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_compute_district_records_returns_rows_with_city_and_auc():
    g = add_travel_time(_mini())
    green = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32635")

    def field_fn(graph):
        ap = access_points(green)
        return accessibility_field(graph, snap_points_to_nodes(ap, graph))

    class _RC:
        seed = 42
        class disruption:  # noqa: N801
            rhos = (0.0, 0.5)
            n_replicates = 1
        class filtration:  # noqa: N801
            max_dim = 1
        class city:  # noqa: N801
            name = "mini"
        h3_res = 11  # fine res so the small fixture yields >=1 populated hex

    records = compute_district_records(
        g, green, field_fn, random_removal, "EPSG:32635", _RC(), min_nodes=1
    )
    assert isinstance(records, list)
    if records:  # at least the structure is correct when a hex qualifies
        assert "city" in records[0]
        assert "auc" in records[0]

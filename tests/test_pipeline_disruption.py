from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, Polygon

from nnj_topology.config import (
    CityConfig,
    DisruptionConfig,
    FiltrationConfig,
    PathsConfig,
    RunConfig,
)
from pipeline.run_disruption import resilience_for_city

FIX = Path(__file__).parent / "fixtures"


def _mini_graph():
    g = nx.MultiDiGraph(nx.read_graphml(FIX / "mini_graph.graphml"))
    for n, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def _rc():
    return RunConfig(
        seed=42,
        h3_res=8,
        city=CityConfig("mini", "Mini", "EPSG:32635"),
        disruption=DisruptionConfig("random", (0.0, 0.5), 2),
        filtration=FiltrationConfig("sublevel", 1),
        paths=PathsConfig("data", "output"),
    )


def test_resilience_for_city_runs_and_increases_or_equal_at_zero():
    g = _mini_graph()
    green = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32635")
    pop = gpd.GeoDataFrame({"population": [1.0]}, geometry=[Point(200, 200)], crs="EPSG:32635")
    res = resilience_for_city(g, green, pop, "EPSG:32635", _rc())
    assert res.distances[0] == 0.0  # no disruption -> zero distance to itself
    assert len(res.distances) == 2

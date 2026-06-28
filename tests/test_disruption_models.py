from pathlib import Path

import networkx as nx

from nnj_topology.disruption import disruption_factory
from nnj_topology.disruption.models import random_removal, targeted_removal

FIX = Path(__file__).parent / "fixtures"


def _mini() -> nx.MultiDiGraph:
    g = nx.read_graphml(FIX / "mini_graph.graphml")
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return nx.MultiDiGraph(g)


def test_random_removal_removes_expected_fraction_and_is_seeded():
    g = _mini()
    m = g.number_of_edges()
    h1 = random_removal(g, rho=0.5, seed=1)
    h2 = random_removal(g, rho=0.5, seed=1)
    assert h1.number_of_edges() == h2.number_of_edges()  # reproducible
    assert h1.number_of_edges() == m - int(0.5 * m)


def test_random_removal_zero_is_identity():
    g = _mini()
    h = random_removal(g, rho=0.0, seed=1)
    assert h.number_of_edges() == g.number_of_edges()


def test_targeted_removal_is_deterministic():
    g = _mini()
    a = targeted_removal(g, rho=0.3)
    b = targeted_removal(g, rho=0.3)
    assert a.number_of_edges() == b.number_of_edges()


def test_factory_dispatch():
    assert disruption_factory("random") is random_removal
    assert disruption_factory("targeted") is targeted_removal

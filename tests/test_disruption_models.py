from pathlib import Path

import networkx as nx
import pytest

from nnj_topology.disruption import disruption_factory
from nnj_topology.disruption.models import (
    hazard_removal,
    random_removal,
    targeted_removal,
)

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
    assert h1.number_of_edges() == h2.number_of_edges()  # reproducible count
    assert h1.number_of_edges() == m - int(0.5 * m)
    # Finding 3: identical edges removed across runs
    assert set(h1.edges(keys=True)) == set(h2.edges(keys=True))


def test_random_removal_zero_is_identity():
    g = _mini()
    h = random_removal(g, rho=0.0, seed=1)
    assert h.number_of_edges() == g.number_of_edges()


def test_targeted_removal_is_deterministic():
    g = _mini()
    a = targeted_removal(g, rho=0.3)
    b = targeted_removal(g, rho=0.3)
    assert a.number_of_edges() == b.number_of_edges()


def test_targeted_removal_removes_exact_directed_fraction():
    # Finding 3: targeted_removal must remove exactly int(rho*m) DIRECTED edges
    g = _mini()
    m = g.number_of_edges()
    h = targeted_removal(g, rho=0.3)
    assert h.number_of_edges() == m - int(0.3 * m)


def test_factory_dispatch():
    assert disruption_factory("random") is random_removal
    assert disruption_factory("targeted") is targeted_removal
    assert disruption_factory("hazard") is hazard_removal


def test_hazard_removal_coverage():
    # Finding 2: hazard_removal correctness and reproducibility
    g = _mini()
    m = g.number_of_edges()
    hazard_nodes = {"0", "1"}  # graphml node ids are strings
    h = hazard_removal(g, rho=0.5, seed=1, hazard_nodes=hazard_nodes)
    # Result has fewer or equal edges vs original
    assert h.number_of_edges() <= m
    # Only edges incident to hazard nodes were removed
    removed_edges = set(g.edges(keys=True)) - set(h.edges(keys=True))
    for u, v, _key in removed_edges:
        assert u in hazard_nodes or v in hazard_nodes, (
            f"Edge ({u}, {v}) was removed but neither endpoint is a hazard node"
        )
    # Reproducible given the same seed
    h2 = hazard_removal(g, rho=0.5, seed=1, hazard_nodes=hazard_nodes)
    assert set(h.edges(keys=True)) == set(h2.edges(keys=True))


def test_factory_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        disruption_factory("nope")


def test_targeted_removal_ranking_reuse_matches():
    """Precomputed ranking must produce identical edge removal as the default path."""
    from nnj_topology.disruption.models import betweenness_ranking

    g = _mini()
    a = targeted_removal(g, 0.3)
    b = targeted_removal(g, 0.3, ranking=betweenness_ranking(g))
    assert set(a.edges(keys=True)) == set(b.edges(keys=True)), (
        "targeted_removal with precomputed ranking removed different edges than default"
    )

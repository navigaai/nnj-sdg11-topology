import networkx as nx
import numpy as np

from nnj_topology.topology.diagrams import (
    essential_finite_split,
    rips_diagram,
    sublevel_diagram,
)
from nnj_topology.topology import filtration_factory


def test_sublevel_diagram_max_dim_zero_skips_h1():
    # A filled triangle: with max_dim=0 we must NOT compute H1 (triangle-fill
    # skipped for speed), only H0.
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    field = {0: 0.0, 1: 0.5, 2: 1.0}
    dgm = sublevel_diagram(g, field, max_dim=0)
    assert set(dgm.keys()) == {0}
    assert 1 not in dgm  # H1 not computed when max_dim=0


def test_sublevel_diagram_handles_large_osm_node_ids():
    # gudhi vertices are 32-bit ints; real OSM node ids exceed 2**31. The
    # sublevel builder must remap ids to a compact index or it raises.
    g = nx.Graph()
    ids = [10_000_000_001, 10_000_000_002, 10_000_000_003, 10_000_000_004]
    nx.add_path(g, ids)
    field = {ids[0]: 0.0, ids[1]: 1.0, ids[2]: 1.0, ids[3]: 0.0}
    dgm = sublevel_diagram(g, field, max_dim=1)
    assert 0 in dgm
    assert dgm[0].shape[1] == 2
    assert dgm[0].shape[0] >= 1  # at least one H0 class


def test_rips_diagram_circle_has_one_h1_class():
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    weights = np.ones(len(pts))
    dgm = rips_diagram(pts, weights, max_dim=1)
    assert 1 in dgm
    # exactly one dominant loop: single longest H1 bar far exceeds any others
    # Robust form: check absolute persistence threshold with margin, then relative dominance
    pers = np.sort(dgm[1][:, 1] - dgm[1][:, 0])[::-1]
    assert len(pers) >= 1
    assert pers[0] > 0.3  # loop persists with healthy margin (empirical ~0.46)
    if len(pers) > 1:
        assert pers[0] > 3 * pers[1]  # longest bar dominates over next-largest


def test_sublevel_diagram_returns_canonical_dict():
    import networkx as nx

    g = nx.path_graph(4)  # 0-1-2-3
    field = {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0}
    dgm = sublevel_diagram(g, field, max_dim=1)
    assert set(dgm.keys()) <= {0, 1}
    assert dgm[0].shape[1] == 2


def test_essential_finite_split_removes_infinities():
    dgm = {0: np.array([[0.0, 1.0], [0.0, np.inf]]), 1: np.array([[0.5, 2.0]])}
    finite, essential = essential_finite_split(dgm)
    assert np.isfinite(finite[0]).all()
    assert finite[0].shape[0] == 1
    assert essential[0].shape[0] == 1


def test_filtration_factory_dispatch():
    assert filtration_factory("rips") is rips_diagram
    assert filtration_factory("sublevel") is sublevel_diagram

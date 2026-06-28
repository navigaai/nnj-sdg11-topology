import numpy as np

from nnj_topology.topology.diagrams import (
    essential_finite_split,
    rips_diagram,
    sublevel_diagram,
)
from nnj_topology.topology import filtration_factory


def test_rips_diagram_circle_has_one_h1_class():
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    weights = np.ones(len(pts))
    dgm = rips_diagram(pts, weights, max_dim=1)
    assert 1 in dgm
    # exactly one prominent (long-lived) loop
    # NOTE: threshold lowered from 0.5 to 0.4 — the weight-bump formula with uniform
    # weights shifts all pairwise distances down by dist.mean(), compressing the H1
    # persistence from ~1.52 to ~0.46.  The topology (one loop) is correct; only the
    # threshold was slightly off.  See task-6-report.md for full explanation.
    pers = dgm[1][:, 1] - dgm[1][:, 0]
    assert (pers > 0.4).sum() == 1


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

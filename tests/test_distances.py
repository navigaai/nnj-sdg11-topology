import numpy as np

from nnj_topology.topology.distances import (
    bottleneck_distance,
    total_persistence,
    wasserstein_distance,
)


def _d(arr):
    return {1: np.array(arr, dtype=float), 0: np.empty((0, 2))}


def test_wasserstein_min_persistence_drops_short_bars():
    # a has a long bar (pers 2.0) + a short noise bar (pers 0.1); b keeps only the long one.
    a = _d([[0.0, 2.0], [0.5, 0.6]])
    b = _d([[0.0, 2.0]])
    # no threshold: the extra short bar in `a` must be matched -> nonzero distance
    assert wasserstein_distance(a, b, dim=1, min_persistence=0.0) > 0.0
    # threshold above the noise bar's lifetime removes it -> diagrams identical
    assert wasserstein_distance(a, b, dim=1, min_persistence=0.5) == 0.0


def test_bottleneck_identity_is_zero():
    a = _d([[0.0, 1.0]])
    assert bottleneck_distance(a, a, dim=1) == 0.0


def test_bottleneck_detects_shift():
    a = _d([[0.0, 1.0]])
    b = _d([[0.0, 2.0]])
    # true bottleneck: matching [0,1]->[0,2] costs Linf = 1.0
    assert abs(bottleneck_distance(a, b, dim=1) - 1.0) < 1e-6


def test_wasserstein_nonnegative_and_symmetric():
    a = _d([[0.0, 1.0], [0.2, 0.5]])
    b = _d([[0.0, 2.0]])
    w_ab = wasserstein_distance(a, b, dim=1)
    w_ba = wasserstein_distance(b, a, dim=1)
    assert w_ab >= 0
    assert abs(w_ab - w_ba) < 1e-6


def test_total_persistence_sums_bar_lengths():
    a = _d([[0.0, 1.0], [0.5, 2.0]])
    assert abs(total_persistence(a, dim=1) - 2.5) < 1e-9

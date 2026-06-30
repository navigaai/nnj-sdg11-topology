from nnj_topology.disruption.resilience import (
    compute_auc,
    critical_rho,
    resilience_curve,
)


def test_compute_auc_linear_ramp():
    rhos = [0.0, 0.5, 1.0]
    dists = [0.0, 0.5, 1.0]
    # trapezoid area = 0.5, normalized by rho-range 1.0 -> 0.5
    assert abs(compute_auc(rhos, dists) - 0.5) < 1e-9


def test_critical_rho_finds_half_max_crossing():
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    dists = [0.0, 0.1, 0.2, 0.9, 1.0]
    # half of max (1.0) is 0.5; first rho where dist >= 0.5 is 0.75
    assert critical_rho(rhos, dists, frac=0.5) == 0.75


def test_critical_rho_none_when_never_crossed():
    rhos = [0.0, 0.5, 1.0]
    dists = [0.0, 0.0, 0.0]
    assert critical_rho(rhos, dists, frac=0.5) is None


def test_resilience_curve_uses_distance_callable():
    res = resilience_curve([0.0, 0.5, 1.0], distance_at_rho=lambda r: r)
    assert res.distances == (0.0, 0.5, 1.0)
    assert abs(res.auc - 0.5) < 1e-9

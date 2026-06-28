"""Finite-safe persistence-diagram distances."""
from __future__ import annotations

import logging

import numpy as np
import persim

from nnj_topology.topology.diagrams import Diagram, essential_finite_split

logger = logging.getLogger(__name__)

__all__ = ["bottleneck_distance", "wasserstein_distance", "total_persistence"]


def _finite(dgm: Diagram, dim: int) -> np.ndarray:
    finite, _ = essential_finite_split(dgm)
    arr = finite.get(dim, np.empty((0, 2)))
    return arr if arr.size else np.empty((0, 2))


def bottleneck_distance(dgm_a: Diagram, dgm_b: Diagram, dim: int = 1) -> float:
    """Bottleneck distance between the finite parts of two diagrams in `dim`.

    Uses the L-inf metric on birth-death coordinates.
    """
    return float(persim.bottleneck(_finite(dgm_a, dim), _finite(dgm_b, dim)))


def wasserstein_distance(
    dgm_a: Diagram, dgm_b: Diagram, dim: int = 1, order: int = 2
) -> float:
    """p-Wasserstein distance (default p=2) between finite parts of two diagrams."""
    return float(persim.wasserstein(_finite(dgm_a, dim), _finite(dgm_b, dim)))


def total_persistence(dgm: Diagram, dim: int = 1) -> float:
    """Sum of bar lengths (death - birth) over finite classes in `dim`."""
    arr = _finite(dgm, dim)
    if arr.size == 0:
        return 0.0
    return float((arr[:, 1] - arr[:, 0]).sum())

"""Resilience curve aggregation: D(rho), AUC, critical rho*."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ResilienceResult", "compute_auc", "critical_rho", "resilience_curve"]


@dataclass(frozen=True)
class ResilienceResult:
    rhos: Tuple[float, ...]
    distances: Tuple[float, ...]
    auc: float
    rho_star: Optional[float]


def compute_auc(rhos: Sequence[float], distances: Sequence[float]) -> float:
    """Trapezoid AUC of D(rho), normalized by the rho range."""
    r = np.asarray(rhos, dtype=float)
    d = np.asarray(distances, dtype=float)
    span = r.max() - r.min()
    if span <= 0:
        return 0.0
    return float(np.trapezoid(d, r) / span)


def critical_rho(
    rhos: Sequence[float], distances: Sequence[float], frac: float = 0.5
) -> Optional[float]:
    """Smallest rho where D(rho) first reaches `frac` of its maximum."""
    d = np.asarray(distances, dtype=float)
    if d.max() <= 0:
        return None
    threshold = frac * d.max()
    for rho, dist in zip(rhos, distances):
        if dist >= threshold:
            return float(rho)
    return None


def resilience_curve(
    rhos: Sequence[float], distance_at_rho: Callable[[float], float]
) -> ResilienceResult:
    """Evaluate D at each rho via the supplied callable and aggregate."""
    rhos_t = tuple(float(r) for r in rhos)
    dists_t = tuple(float(distance_at_rho(r)) for r in rhos_t)
    return ResilienceResult(
        rhos=rhos_t,
        distances=dists_t,
        auc=compute_auc(rhos_t, dists_t),
        rho_star=critical_rho(rhos_t, dists_t),
    )

"""Disruption scenarios + registry."""
import logging
from typing import Callable, Dict

from nnj_topology.disruption.models import (
    hazard_removal,
    random_removal,
    targeted_removal,
)
from nnj_topology.disruption.resilience import (
    ResilienceResult,
    compute_auc,
    critical_rho,
    resilience_curve,
)

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Callable] = {
    "random": random_removal,
    "targeted": targeted_removal,
    "hazard": hazard_removal,
}

__all__ = [
    "random_removal",
    "targeted_removal",
    "hazard_removal",
    "register_disruption",
    "disruption_factory",
    "ResilienceResult",
    "compute_auc",
    "critical_rho",
    "resilience_curve",
]


def register_disruption(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


def disruption_factory(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown disruption '{name}'; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]

"""Topology: persistent homology constructions, diagrams, distances."""
from typing import Callable, Dict

from nnj_topology.topology.diagrams import (
    Diagram,
    essential_finite_split,
    rips_diagram,
    sublevel_diagram,
)
from nnj_topology.topology.distances import (
    bottleneck_distance,
    total_persistence,
    wasserstein_distance,
)
from nnj_topology.topology.filtration import BUILTIN_FILTRATIONS

_REGISTRY: Dict[str, Callable] = dict(BUILTIN_FILTRATIONS)

__all__ = [
    "Diagram",
    "rips_diagram",
    "sublevel_diagram",
    "essential_finite_split",
    "register_filtration",
    "filtration_factory",
    "bottleneck_distance",
    "wasserstein_distance",
    "total_persistence",
]


def register_filtration(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


def filtration_factory(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown filtration '{name}'; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]

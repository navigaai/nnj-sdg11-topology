"""Filtration registry wiring (keeps construction selection config-driven)."""
from __future__ import annotations

from nnj_topology.topology.diagrams import rips_diagram, sublevel_diagram

__all__ = ["BUILTIN_FILTRATIONS"]

BUILTIN_FILTRATIONS = {"rips": rips_diagram, "sublevel": sublevel_diagram}

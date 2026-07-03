"""Persistence diagram computation (Rips and sublevel-set)."""
from __future__ import annotations

import logging
from typing import Dict

import gudhi
import networkx as nx
import numpy as np
from ripser import ripser

logger = logging.getLogger(__name__)

Diagram = Dict[int, np.ndarray]

__all__ = ["Diagram", "rips_diagram", "sublevel_diagram", "essential_finite_split"]


def rips_diagram(points_xy: np.ndarray, weights: np.ndarray, max_dim: int = 1) -> Diagram:
    """Weighted Vietoris-Rips persistence on 2-D access points.

    `weights` (e.g. population) scale point radii so that densely demanded
    coverage holes persist longer. Implemented as a weighted-Rips lower star
    via ripser's `distance_matrix` with additive weight offsets.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    w = weights / (weights.max() + 1e-12)
    # higher weight -> earlier birth: subtract a scaled weight bump, clip >= 0
    bump = (w[:, None] + w[None, :]) * 0.5
    dist = np.clip(dist - bump * dist.mean(), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    res = ripser(dist, distance_matrix=True, maxdim=max_dim)
    return {d: np.atleast_2d(res["dgms"][d]) if res["dgms"][d].size else np.empty((0, 2))
            for d in range(max_dim + 1)}


def sublevel_diagram(graph: nx.Graph, field: dict, max_dim: int = 1) -> Diagram:
    """Sublevel-set persistence of a node field on a graph (1-skeleton + filled triangles).

    Node ids are remapped to a compact 0..N-1 index before insertion: gudhi
    vertices are 32-bit ints, whereas real OSM node ids exceed 2**31. The
    remapping is label-invariant, so the persistence values are unchanged.
    """
    st = gudhi.SimplexTree()
    idx = {node: i for i, node in enumerate(field)}
    for node, value in field.items():
        st.insert([idx[node]], filtration=float(value))
    for u, v in graph.edges():
        fu, fv = float(field[u]), float(field[v])
        st.insert([idx[u], idx[v]], filtration=max(fu, fv))
    # Fill triangles on every 3-clique so H1 reflects genuine enclosed voids.
    # Triangles do not affect H0 (connected components), so this expensive
    # enumeration is skipped for an H0-only computation (max_dim == 0).
    if max_dim >= 1:
        for clique in nx.enumerate_all_cliques(nx.Graph(graph)):
            if len(clique) == 3:
                vals = [float(field[c]) for c in clique]
                st.insert([idx[c] for c in clique], filtration=max(vals))
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    out: Diagram = {d: [] for d in range(max_dim + 1)}
    for dim, (birth, death) in st.persistence():
        if dim <= max_dim:
            out[dim].append([birth, death])
    return {d: (np.array(v) if v else np.empty((0, 2))) for d, v in out.items()}


def essential_finite_split(dgm: Diagram) -> tuple[Diagram, Diagram]:
    """Split each dimension's diagram into finite-death and infinite-death parts."""
    finite: Diagram = {}
    essential: Diagram = {}
    for dim, arr in dgm.items():
        if arr.size == 0:
            finite[dim] = np.empty((0, 2))
            essential[dim] = np.empty((0, 2))
            continue
        is_inf = ~np.isfinite(arr[:, 1])
        finite[dim] = arr[~is_inf]
        essential[dim] = arr[is_inf]
    return finite, essential

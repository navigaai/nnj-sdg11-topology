"""Extended-persistence H0 diagram for a node field on a graph (helper).

Uses GUDHI extended persistence so that the essential H0 classes (whole connected
components, including capped 'unreachable' pockets) receive FINITE (birth, death)
coordinates in a separate subdiagram, instead of dropping out at infinity. This
keeps two channels distinct:
  - Ordinary H0  : merging of well-served sub-regions (the graded access signal)
  - Extended H0  : connected-component structure (the disconnection signal)
"""
from __future__ import annotations

import gudhi
import networkx as nx
import numpy as np


def extended_h0(sub: nx.Graph, field: dict, cap: float = 60.0,
                min_persistence: float = 1.0) -> np.ndarray:
    """Return an (N,2) array of oriented (birth<=death) H0 points from extended
    persistence of the sublevel filtration of ``field`` on ``sub``.

    Unreachable nodes (inf) are capped at ``cap`` so they enter the filtration
    finitely and appear as extended (component) features rather than vanishing.
    Points with lifetime < ``min_persistence`` are dropped (denoising).
    """
    nodes = list(sub.nodes)
    if not nodes:
        return np.empty((0, 2))
    idx = {n: i for i, n in enumerate(nodes)}
    st = gudhi.SimplexTree()
    for n in nodes:
        v = field[n]
        st.insert([idx[n]], filtration=float(v if np.isfinite(v) else cap))
    for u, v in sub.edges():
        fu = field[u] if np.isfinite(field[u]) else cap
        fv = field[v] if np.isfinite(field[v]) else cap
        st.insert([idx[u], idx[v]], filtration=float(max(fu, fv)))
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    dgms = st.extended_persistence(min_persistence=0.0)
    pts = []
    for sub_d in dgms:  # Ordinary, Relative, Ext+, Ext-
        for dim, (b, d) in sub_d:
            if dim == 0 and np.isfinite(b) and np.isfinite(d):
                lo, hi = (b, d) if b <= d else (d, b)
                if hi - lo >= min_persistence:
                    pts.append([lo, hi])
    return np.array(pts) if pts else np.empty((0, 2))

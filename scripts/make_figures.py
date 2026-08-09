"""Generate additional manuscript figures from committed real artifacts.

Usage: uv run python scripts/make_figures.py [robustness|persistence|accessfield|all]
Outputs vector PDFs to output/figures/ and paper/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("output/figures")
PAPER = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colourblind-safe palette
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

DESC = ["circuity", "orientation_entropy", "mean_street_length", "intersection_density"]
DESC_LABEL = {
    "circuity": "circuity",
    "orientation_entropy": "orientation\nentropy",
    "mean_street_length": "mean street\nlength",
    "intersection_density": "intersection\ndensity",
}


def _save(fig, name):
    for d in (OUT, PAPER):
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf")


def robustness():
    """Two panels: (A) coefficients across 3 disruption scenarios (res 8, from CSVs);
    (B) sign+significance grid across H3 resolutions 7/8/9 (recorded in RESULTS.md)."""
    scen = {s: pd.read_csv(f"output/regression_{s}.csv").set_index("term")
            for s in ("random", "targeted")}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 4.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # Panel A: forest plot, coef +/- 1.96 se, per descriptor per scenario
    ypos = np.arange(len(DESC))[::-1]
    offs = {"random": +0.15, "targeted": -0.15}
    col = {"random": CB[0], "targeted": CB[1]}
    for s in ("random", "targeted"):
        df = scen[s]
        xs = [df.loc[d, "coef"] for d in DESC]
        es = [1.96 * df.loc[d, "std_err"] for d in DESC]
        axA.errorbar(xs, ypos + offs[s], xerr=es, fmt="o", color=col[s],
                     capsize=3, ms=5, label=s, lw=1.5)
    axA.axvline(0, color="0.5", lw=1, ls="--")
    axA.set_yticks(ypos)
    axA.set_yticklabels([DESC_LABEL[d] for d in DESC])
    axA.set_xlabel(r"coefficient on AUC (higher AUC = less resilient)")
    axA.set_title("(a) Across distributed disruption scenarios (H3 res 8)")
    axA.legend(frameon=False, fontsize=8, loc="lower right")

    # Panel B: sign x significance grid across resolutions (recorded values).
    # Rows = descriptors, cols = res 7/8/9. Marker: filled = p<0.001, open = p<0.05, x = ns.
    # Colour: blue = negative coef, red = positive.
    res = {  # (coef_sign, sig_level) ; sig: 3=p<.001, 2=p<.05, 0=ns
        "circuity":            {7: (-1, 3), 8: (-1, 3), 9: (-1, 3)},
        "orientation_entropy": {7: (+1, 3), 8: (+1, 3), 9: (+1, 3)},
        "mean_street_length":  {7: (-1, 3), 8: (-1, 3), 9: (-1, 3)},
        # Intersection density is the weakest, grid-sensitive effect: not
        # significant at the headline full 8x10 grid (res 8, p=0.22); shown ns.
        "intersection_density":{7: (-1, 0), 8: (-1, 0), 9: (-1, 0)},
    }
    cols = [7, 8, 9]
    for i, d in enumerate(DESC):
        y = ypos[i]
        for j, r in enumerate(cols):
            sign, sig = res[d][r]
            c = CB[0] if sign < 0 else CB[1]
            if sig == 3:
                axB.plot(j, y, "o", color=c, ms=13)
            elif sig == 2:
                axB.plot(j, y, "o", mfc="white", mec=c, ms=13, mew=2)
            else:
                axB.plot(j, y, "x", color="0.6", ms=10, mew=2)
    axB.set_xticks(range(len(cols)))
    axB.set_xticklabels([f"res {r}" for r in cols])
    axB.set_yticks(ypos)
    axB.set_yticklabels([])
    axB.set_xlim(-0.5, len(cols) - 0.5)
    axB.set_ylim(-0.6, len(DESC) - 0.4)
    axB.set_title("(b) Across H3 resolutions")
    # legend
    from matplotlib.lines import Line2D
    leg = [
        Line2D([], [], marker="o", color=CB[0], ls="", ms=10, label="neg., $p<0.001$"),
        Line2D([], [], marker="o", mfc="white", mec=CB[0], ls="", ms=10, mew=2, label="$p<0.05$"),
        Line2D([], [], marker="x", color="0.6", ls="", ms=9, mew=2, label="n.s."),
    ]
    axB.legend(handles=leg, frameon=False, fontsize=7.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.12), ncol=3)

    fig.suptitle("Morphology$\\rightarrow$resilience coefficients are stable across "
                 "disruption model and district resolution", fontsize=11, y=1.02)
    _save(fig, "fig_robustness")


def persistence():
    """H0 persistence diagram of the Amsterdam baseline field, contrasting the rich
    finite H0 with the near-empty finite H1 (motivates the H0 metric choice)."""
    d = np.load("output/amsterdam/baseline_diagram.npz")
    h0 = d["dim0"]
    h1 = d["dim1"] if "dim1" in d.files else np.empty((0, 2))
    h0f = h0[np.isfinite(h0[:, 1])]
    h1f = h1[np.isfinite(h1[:, 1])] if h1.size else np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(5.2, 5))
    top = float(h0f[:, 1].max()) if h0f.size else 30
    ax.plot([0, top], [0, top], color="0.6", lw=1, ls="--", zorder=0)
    ax.scatter(h0f[:, 0], h0f[:, 1], s=16, alpha=0.5, color=CB[0],
               label=f"$H_0$ (finite): {h0f.shape[0]}")
    if h1f.size:
        ax.scatter(h1f[:, 0], h1f[:, 1], s=40, color=CB[1], marker="^",
                   label=f"$H_1$ (finite): {h1f.shape[0]}")
    else:
        ax.scatter([], [], color=CB[1], marker="^", label="$H_1$ (finite): 0")
    ax.set_xlabel("birth (walk-minutes to nearest green space)")
    ax.set_ylabel("death (walk-minutes)")
    ax.set_title("Amsterdam baseline persistence diagram\n"
                 "the resilience signal lives in $H_0$", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    _save(fig, "fig_persistence")


def accessfield():
    """Accessibility field heatmap: Amsterdam walk network nodes coloured by
    walk-time to the nearest green space (clipped to the GHS-UCDB urban centre)."""
    from nnj_topology.accessibility.field import accessibility_field, add_travel_time
    from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
    from nnj_topology.data.greenspace import access_points, load_greenspace, snap_points_to_nodes
    from nnj_topology.data.network import load_walk_network

    crs = "EPSG:28992"
    g = load_walk_network("Amsterdam, Netherlands", crs, Path("data/amsterdam/walk.graphml"))
    g = clip_graph_to_boundary(
        g, load_urban_boundary(Path("data/ghsl/ghs_ucdb.gpkg"), "amsterdam", crs), crs)
    g = add_travel_time(g)
    green = load_greenspace("Amsterdam, Netherlands", crs, Path("data/amsterdam/green.gpkg"))
    field = accessibility_field(g, snap_points_to_nodes(access_points(green), g))

    xs = np.array([float(g.nodes[n]["x"]) for n in g.nodes])
    ys = np.array([float(g.nodes[n]["y"]) for n in g.nodes])
    vals = np.array([field[n] for n in g.nodes])
    finite = np.isfinite(vals)
    vals = np.clip(vals, 0, np.nanpercentile(vals[finite], 98))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sc = ax.scatter(xs[finite], ys[finite], c=vals[finite], s=1.5,
                    cmap="RdYlGn_r", linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("walk-time to nearest green space (min)")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Amsterdam: green-space accessibility field\n"
                 f"({finite.sum():,} network nodes, GHS-UCDB urban centre)", fontsize=10)
    _save(fig, "fig_accessfield")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("robustness", "all"):
        robustness()
    if which in ("persistence", "all"):
        persistence()
    if which in ("accessfield", "all"):
        accessfield()

"""Manuscript figures (return Figure objects; caller saves)."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from nnj_topology.disruption.resilience import ResilienceResult
from nnj_topology.topology.diagrams import Diagram

logger = logging.getLogger(__name__)

__all__ = [
    "plot_persistence_diagram",
    "plot_resilience_curves",
    "plot_morphology_vs_resilience",
]


def plot_persistence_diagram(dgm: Diagram, dim: int = 1) -> Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    arr = dgm.get(dim, np.empty((0, 2)))
    finite = arr[np.isfinite(arr[:, 1])] if arr.size else arr
    if finite.size:
        ax.scatter(finite[:, 0], finite[:, 1], s=20, alpha=0.7)
        top = float(finite.max())
        ax.plot([0, top], [0, top], "k--", lw=0.8)
    ax.set_xlabel("birth (walk minutes)")
    ax.set_ylabel("death (walk minutes)")
    ax.set_title(f"H{dim} persistence diagram")
    fig.tight_layout()
    return fig


def plot_resilience_curves(results: dict[str, ResilienceResult]) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    for city, res in results.items():
        ax.plot(res.rhos, res.distances, marker="o", label=city)
        if res.rho_star is not None:
            ax.axvline(res.rho_star, ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"disruption intensity $\rho$")
    ax.set_ylabel(r"diagram distance $D(\rho)$")
    ax.set_title("Resilience curves")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_morphology_vs_resilience(
    df: pd.DataFrame, feature: str, target: str = "auc"
) -> Figure:
    """Headline Fig. 6: districts (points) coloured by city, with a pooled fit line."""
    fig, ax = plt.subplots(figsize=(5, 4))
    if "city" in df.columns:
        for city, grp in df.groupby("city"):
            ax.scatter(grp[feature], grp[target], s=12, alpha=0.5, label=str(city))
        ax.legend(fontsize=7, title="city")
    else:
        ax.scatter(df[feature], df[target], s=12, alpha=0.5)
    # pooled least-squares trend line
    if len(df) >= 2:
        coef = np.polyfit(df[feature], df[target], 1)
        xs = np.linspace(df[feature].min(), df[feature].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "k--", lw=1.0)
    ax.set_xlabel(feature.replace("_", " "))
    ax.set_ylabel(target)
    ax.set_title(f"District morphology vs. resilience ({feature})")
    fig.tight_layout()
    return fig

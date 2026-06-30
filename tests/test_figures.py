import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from nnj_topology.disruption.resilience import ResilienceResult  # noqa: E402
from nnj_topology.viz.figures import (  # noqa: E402
    plot_morphology_vs_resilience,
    plot_persistence_diagram,
    plot_resilience_curves,
)


def test_plot_persistence_diagram_returns_figure():
    dgm = {0: np.array([[0.0, 1.0]]), 1: np.array([[0.2, 0.9]])}
    fig = plot_persistence_diagram(dgm, dim=1)
    assert isinstance(fig, Figure)


def test_plot_resilience_curves_returns_figure():
    res = {"a": ResilienceResult((0.0, 1.0), (0.0, 1.0), 0.5, 0.5)}
    fig = plot_resilience_curves(res)
    assert isinstance(fig, Figure)


def test_plot_morphology_vs_resilience_returns_figure():
    df = pd.DataFrame({"city": ["a", "b"], "circuity": [1.0, 1.2], "auc": [0.1, 0.2]})
    fig = plot_morphology_vs_resilience(df, feature="circuity", target="auc")
    assert isinstance(fig, Figure)

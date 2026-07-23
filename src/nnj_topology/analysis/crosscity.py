"""Descriptive city typology overlay (NOT used for inference)."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["city_typology"]


def city_typology(df: pd.DataFrame) -> pd.DataFrame:
    """Per-city descriptive means of district resilience summaries."""
    agg = (
        df.groupby("city")
        .agg(
            auc_mean=("auc", "mean"),
            rho_star_mean=("rho_star", "mean"),
            total_persistence_mean=("total_persistence", "mean"),
            n_districts=("auc", "size"),
        )
        .reset_index()
        .sort_values("city")
        .reset_index(drop=True)
    )
    return agg

"""District-level inferential analysis (the C3 claim)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import statsmodels.formula.api as smf

if TYPE_CHECKING:
    import statsmodels.regression.linear_model

logger = logging.getLogger(__name__)

__all__ = [
    "build_district_frame",
    "fixed_effects_regression",
    "tidy_coefficients",
    "city_clustered_regression",
]

_NON_FEATURE = {"city", "auc", "rho_star", "total_persistence", "hex"}


def build_district_frame(records: List[dict]) -> pd.DataFrame:
    """Assemble one row per district from local resilience + morphology records."""
    return pd.DataFrame(records).reset_index(drop=True)


def fixed_effects_regression(
    df: pd.DataFrame, target: str = "auc", features: Optional[List[str]] = None
) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":
    """OLS of `target` on morphology features with city fixed effects."""
    if features is None:
        features = [c for c in df.columns if c not in _NON_FEATURE]
    rhs = " + ".join(features + ["C(city)"])
    formula = f"{target} ~ {rhs}"
    logger.info("Fitting: %s", formula)
    return smf.ols(formula, data=df).fit()


def city_clustered_regression(
    df: pd.DataFrame, target: str = "auc", features: Optional[List[str]] = None
) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":
    """Same fixed-effects OLS, but with standard errors clustered by city.

    Districts within a city are not independent draws: they share the same
    street network, climate and planning regime, so classical OLS standard
    errors understate uncertainty. Clustering by city (the level at which the
    disruption is applied) yields inference that is honest about the fact that
    the design has five clusters, not 2,603 independent units.
    """
    if features is None:
        features = [c for c in df.columns if c not in _NON_FEATURE]
    rhs = " + ".join(features + ["C(city)"])
    formula = f"{target} ~ {rhs}"
    logger.info("Fitting (city-clustered SE): %s", formula)
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["city"]}
    )


def tidy_coefficients(result: "statsmodels.regression.linear_model.RegressionResultsWrapper") -> pd.DataFrame:
    """Tidy morphology coefficients (drop the city fixed-effect dummies)."""
    params = result.params
    pvalues = result.pvalues
    bse = result.bse
    rows = []
    for term in params.index:
        if term.startswith("C(city)") or term == "Intercept":
            continue
        rows.append(
            {
                "term": term,
                "coef": float(params[term]),
                "std_err": float(bse[term]),
                "p_value": float(pvalues[term]),
            }
        )
    return pd.DataFrame(rows)

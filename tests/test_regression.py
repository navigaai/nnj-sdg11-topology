import numpy as np
import pandas as pd

from nnj_topology.analysis.regression import (
    build_district_frame,
    fixed_effects_regression,
    tidy_coefficients,
)


def _district_records(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        city = ["A", "B", "C"][i % 3]
        circuity = rng.normal(1.1, 0.1)
        # auc depends on circuity (+) plus a city offset + noise
        offset = {"A": 0.0, "B": 0.2, "C": 0.4}[city]
        auc = 0.8 * circuity + offset + rng.normal(0, 0.02)
        rows.append({"city": city, "circuity": circuity, "intersection_density": rng.normal(50, 5),
                     "auc": auc, "rho_star": rng.uniform(0.2, 0.5), "total_persistence": rng.uniform(1, 5)})
    return rows


def test_build_district_frame_keeps_city_column():
    df = build_district_frame(_district_records(30))
    assert "city" in df.columns
    assert len(df) == 30


def test_fixed_effects_recovers_positive_circuity_effect():
    df = build_district_frame(_district_records(300))
    result = fixed_effects_regression(df, target="auc", features=["circuity", "intersection_density"])
    tidy = tidy_coefficients(result)
    row = tidy[tidy["term"] == "circuity"].iloc[0]
    assert row["coef"] > 0          # true effect is +0.8
    assert row["p_value"] < 0.05    # well-powered at n=300

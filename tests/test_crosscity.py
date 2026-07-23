import pandas as pd

from nnj_topology.analysis.crosscity import city_typology


def test_city_typology_one_row_per_city():
    df = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "auc": [0.1, 0.3, 0.5, 0.7],
            "rho_star": [0.4, 0.4, 0.3, 0.3],
            "total_persistence": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = city_typology(df)
    assert list(out["city"]) == ["A", "B"]
    assert abs(out.loc[out["city"] == "A", "auc_mean"].iloc[0] - 0.2) < 1e-9

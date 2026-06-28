from pathlib import Path

from nnj_topology.data.population import load_population_points

FIX = Path(__file__).parent / "fixtures"


def test_load_population_points_skips_empty_cells():
    pts = load_population_points(FIX / "mini_pop.tif", crs="EPSG:32635", threshold=1.0)
    # 5 nonzero cells in the fixture
    assert len(pts) == 5
    assert "population" in pts.columns
    assert pts["population"].min() >= 1.0
    assert pts["population"].sum() == 46.0  # 10+5+20+3+8

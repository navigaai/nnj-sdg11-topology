import geopandas as gpd
from shapely.geometry import Polygon

from nnj_topology.morphology.descriptors import greenspace_fragmentation


def test_fragmentation_higher_for_many_small_patches():
    one_big = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])], crs="EPSG:32635"
    )
    many_small = gpd.GeoDataFrame(
        geometry=[
            Polygon([(i, 0), (i + 10, 0), (i + 10, 10), (i, 10)]) for i in range(0, 100, 20)
        ],
        crs="EPSG:32635",
    )
    assert greenspace_fragmentation(many_small) > greenspace_fragmentation(one_big)


def test_fragmentation_zero_area_returns_zero():
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32635")
    assert greenspace_fragmentation(empty) == 0.0

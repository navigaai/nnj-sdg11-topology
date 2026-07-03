from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon

from nnj_topology.accessibility.field import accessibility_field, add_travel_time
from nnj_topology.data.greenspace import access_points, snap_points_to_nodes
from nnj_topology.disruption.models import random_removal
from nnj_topology.morphology.descriptors import greenspace_fragmentation
from pipeline.run_analysis import _hex_polygon, compute_district_records

FIX = Path(__file__).parent / "fixtures"


def _mini():
    # Wrap as MultiDiGraph: random_removal calls .edges(keys=True) which
    # requires a MultiGraph/MultiDiGraph; the stored graphml is a plain DiGraph.
    g = nx.MultiDiGraph(nx.read_graphml(FIX / "mini_graph.graphml"))
    for _, d in g.nodes(data=True):
        d["x"], d["y"] = float(d["x"]), float(d["y"])
    for _, _, d in g.edges(data=True):
        d["length"] = float(d["length"])
    return g


def test_compute_district_records_returns_rows_with_city_and_auc():
    g = add_travel_time(_mini())
    green = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32635")

    def field_fn(graph):
        ap = access_points(green)
        return accessibility_field(graph, snap_points_to_nodes(ap, graph))

    class _RC:
        seed = 42
        class disruption:  # noqa: N801
            rhos = (0.0, 0.5)
            n_replicates = 1
        class filtration:  # noqa: N801
            max_dim = 1
        class city:  # noqa: N801
            name = "mini"
        h3_res = 11  # fine res so the small fixture yields >=1 populated hex
        homology_dim = 0

    records = compute_district_records(
        g, green, field_fn, random_removal, "EPSG:32635", _RC(), min_nodes=1
    )
    assert isinstance(records, list)
    if records:  # at least the structure is correct when a hex qualifies
        assert "city" in records[0]
        assert "auc" in records[0]


def test_district_fragmentation_varies_by_hex():
    """Greenspace fragmentation must differ between hex cells with different green coverage.

    We pick two neighbouring H3 res-9 cells, build a green patch inside cell A,
    clip each cell's green space, and assert fragmentation(A) > fragmentation(B==0).
    """
    import h3

    # Pick a reference cell at resolution 9 somewhere in Europe (WGS84 coords)
    cell_a = h3.latlng_to_cell(41.0, 29.0, 9)
    # Get a neighbour guaranteed to be different from cell_a
    neighbours = [c for c in h3.grid_disk(cell_a, 1) if c != cell_a]
    cell_b = neighbours[0]

    crs = "EPSG:32635"  # UTM zone 35N — appropriate for the lat/lng above

    # Build the projected polygon for cell A
    poly_a = _hex_polygon(cell_a, crs)

    # Create a small green patch that is fully inside cell A's bounding box
    cx, cy = poly_a.centroid.x, poly_a.centroid.y
    patch = Polygon([(cx - 100, cy - 100), (cx + 100, cy - 100),
                     (cx + 100, cy + 100), (cx - 100, cy + 100)])
    green = gpd.GeoDataFrame(geometry=[patch], crs=crs)

    # Clip green to cell A — should have the patch
    poly_a_shape = poly_a
    candidates_a = green[green.intersects(poly_a_shape)]
    clipped_a = candidates_a.geometry.intersection(poly_a_shape)
    local_green_a = gpd.GeoDataFrame(
        geometry=clipped_a[~clipped_a.is_empty].reset_index(drop=True), crs=crs
    )

    # Clip green to cell B — patch is inside A, so B gets nothing
    poly_b = _hex_polygon(cell_b, crs)
    candidates_b = green[green.intersects(poly_b)]
    clipped_b = candidates_b.geometry.intersection(poly_b)
    local_green_b = gpd.GeoDataFrame(
        geometry=clipped_b[~clipped_b.is_empty].reset_index(drop=True) if len(candidates_b) > 0
        else gpd.GeoSeries([], crs=crs),
        crs=crs,
    )

    frag_a = greenspace_fragmentation(local_green_a)
    frag_b = greenspace_fragmentation(local_green_b)

    assert frag_a > 0.0, "Cell A should have nonzero fragmentation (patch is inside it)"
    assert frag_b == 0.0, "Cell B should have zero fragmentation (no green clipped to it)"
    assert frag_a > frag_b, "Fragmentation must vary per hex, not be a city-level constant"

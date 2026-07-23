"""Regenerate committed test fixtures. Run manually: uv run python tests/fixtures/make_fixtures.py"""
from pathlib import Path

import networkx as nx

FIX = Path(__file__).parent


def make_mini_graph() -> None:
    # 3x3 grid of nodes with metric x/y coords and edge length attrs.
    G = nx.MultiDiGraph(crs="EPSG:32635")
    coords = {i * 3 + j: (j * 100.0, i * 100.0) for i in range(3) for j in range(3)}
    for n, (x, y) in coords.items():
        G.add_node(n, x=x, y=y)
    for i in range(3):
        for j in range(3):
            n = i * 3 + j
            if j < 2:
                m = n + 1
                G.add_edge(n, m, length=100.0)
                G.add_edge(m, n, length=100.0)
            if i < 2:
                m = n + 3
                G.add_edge(n, m, length=100.0)
                G.add_edge(m, n, length=100.0)
    nx.write_graphml(G, FIX / "mini_graph.graphml")


def make_mini_pop() -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    arr = np.array([[10.0, 0.0, 5.0], [0.0, 20.0, 0.0], [3.0, 0.0, 8.0]], dtype="float32")
    transform = from_origin(0.0, 300.0, 100.0, 100.0)  # 100m cells, origin top-left
    with rasterio.open(
        FIX / "mini_pop.tif", "w", driver="GTiff", height=3, width=3, count=1,
        dtype="float32", crs="EPSG:32635", transform=transform,
    ) as dst:
        dst.write(arr, 1)


def make_mini_dem() -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    arr = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]], dtype="float32")
    transform = from_origin(0.0, 300.0, 100.0, 100.0)
    with rasterio.open(
        FIX / "mini_dem.tif", "w", driver="GTiff", height=3, width=3, count=1,
        dtype="float32", crs="EPSG:32635", transform=transform,
    ) as dst:
        dst.write(arr, 1)


if __name__ == "__main__":
    make_mini_graph()
    make_mini_pop()
    make_mini_dem()
    print("fixtures written")

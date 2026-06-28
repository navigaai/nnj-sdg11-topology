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


if __name__ == "__main__":
    make_mini_graph()
    print("fixtures written")

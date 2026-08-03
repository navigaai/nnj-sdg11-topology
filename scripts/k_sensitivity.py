"""Betweenness-k sensitivity (reviewer response).

The targeted-disruption scenario ranks edges by betweenness centrality
approximated from k sampled pivot sources (headline k=500). A reviewer asked
whether the ranking---and therefore which edges the scenario removes---is stable
in k. Rather than re-run the whole pipeline at several k (which changes only the
edge ordering), we test the ordering directly: for each city we compute the
betweenness ranking at k in {250, 500, 1000} and measure

  * Spearman rank correlation between the k=500 ranking and the k=250 / k=1000
    rankings over the shared edge set, and
  * Jaccard overlap of the top-decile (highest-betweenness) edge set, which is
    the part of the ranking the disruption actually consumes.

High values mean the targeted scenario is insensitive to the pivot count.

Usage: uv run python scripts/k_sensitivity.py
Writes output/k_sensitivity.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from nnj_topology.data.boundary import clip_graph_to_boundary, load_urban_boundary
from nnj_topology.data.network import load_walk_network
from nnj_topology.disruption.models import betweenness_ranking

# Ranking stability is verified on the Amsterdam urban-centre network (the
# smallest by node count, ~49k nodes), where betweenness at k up to 1000 pivots is
# tractable; the sampling approximation is a property of the estimator, so a
# representative check on one network is informative. (Exact betweenness is
# O(V*E) and intractable at city scale, which is why the pivot approximation is
# used at all.)
CITIES = {
    "amsterdam": ("Amsterdam, Netherlands", "EPSG:28992"),
}
KS = [250, 500, 1000]


def _rank_map(ranking: list) -> dict:
    """edge -> rank position (0 = highest betweenness)."""
    return {e: i for i, e in enumerate(ranking)}


def _jaccard_top(a: list, b: list, frac: float = 0.1) -> float:
    n = max(1, int(round(frac * min(len(a), len(b)))))
    sa, sb = set(a[:n]), set(b[:n])
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    rows = []
    for city, (place, crs) in CITIES.items():
        g = load_walk_network(place, crs, Path(f"data/{city}/walk.graphml"))
        boundary = load_urban_boundary(Path("data/ghsl/ghs_ucdb.gpkg"), city, crs)
        g = clip_graph_to_boundary(g, boundary, crs)
        rankings = {k: betweenness_ranking(g, k=k) for k in KS}
        ref = rankings[500]
        ref_rank = _rank_map(ref)
        for k in (250, 1000):
            rk = _rank_map(rankings[k])
            common = [e for e in ref_rank if e in rk]
            rho = spearmanr(
                [ref_rank[e] for e in common], [rk[e] for e in common]
            ).correlation
            rows.append(
                {
                    "city": city,
                    "k_vs_500": k,
                    "n_edges": len(ref),
                    "spearman": round(float(rho), 4),
                    "jaccard_top10pct": round(_jaccard_top(ref, rankings[k]), 4),
                }
            )
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv("output/k_sensitivity.csv", index=False)
    print("\nSummary (mean over cities):")
    print(df.groupby("k_vs_500")[["spearman", "jaccard_top10pct"]].mean().round(3))


if __name__ == "__main__":
    main()

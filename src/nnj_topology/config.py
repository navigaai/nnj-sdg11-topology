"""Frozen config dataclasses mirroring the Hydra schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import DictConfig

__all__ = [
    "CityConfig",
    "DisruptionConfig",
    "FiltrationConfig",
    "PathsConfig",
    "RunConfig",
    "from_omegaconf",
]


@dataclass(frozen=True)
class CityConfig:
    name: str
    place: str
    crs: str


@dataclass(frozen=True)
class DisruptionConfig:
    name: str
    rhos: Tuple[float, ...]
    n_replicates: int


@dataclass(frozen=True)
class FiltrationConfig:
    name: str
    max_dim: int


@dataclass(frozen=True)
class PathsConfig:
    data: str
    output: str


@dataclass(frozen=True)
class RunConfig:
    seed: int
    h3_res: int
    # Homology dimension for the resilience distance. Default 0 (H0): on real
    # street networks the sublevel filtration's H1 is near-empty (few triangles
    # fill the graph's cycles), so the resilience signal lives in H0 (merging of
    # well-served regions). See RESULTS.md / methods.
    homology_dim: int
    # Topological-denoising threshold (walk-minutes): H0 classes with lifetime
    # below this are dropped before the Wasserstein match. Keeps the exact metric
    # while bounding its O(n^3) cost on city-scale diagrams. 0.0 = no denoising.
    persistence_threshold: float
    city: CityConfig
    disruption: DisruptionConfig
    filtration: FiltrationConfig
    paths: PathsConfig
    # Number of pivot sources for sampled edge-betweenness in the targeted
    # scenario. None -> default min(500, V). Exposed for k-sensitivity analysis.
    k_pivots: Optional[int] = None


def from_omegaconf(cfg: DictConfig) -> RunConfig:
    """Convert a resolved OmegaConf config into a frozen RunConfig."""
    return RunConfig(
        seed=int(cfg.seed),
        h3_res=int(cfg.h3_res),
        homology_dim=int(cfg.get("homology_dim", 0)),
        persistence_threshold=float(cfg.get("persistence_threshold", 1.0)),
        city=CityConfig(name=cfg.city.name, place=cfg.city.place, crs=cfg.city.crs),
        disruption=DisruptionConfig(
            name=cfg.disruption.name,
            rhos=tuple(float(r) for r in cfg.disruption.rhos),
            n_replicates=int(cfg.disruption.n_replicates),
        ),
        filtration=FiltrationConfig(name=cfg.filtration.name, max_dim=int(cfg.filtration.max_dim)),
        paths=PathsConfig(data=cfg.paths.data, output=cfg.paths.output),
        k_pivots=(int(cfg.k_pivots) if cfg.get("k_pivots", None) is not None else None),
    )

"""Frozen config dataclasses mirroring the Hydra schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
    city: CityConfig
    disruption: DisruptionConfig
    filtration: FiltrationConfig
    paths: PathsConfig


def from_omegaconf(cfg: DictConfig) -> RunConfig:
    """Convert a resolved OmegaConf config into a frozen RunConfig."""
    return RunConfig(
        seed=int(cfg.seed),
        h3_res=int(cfg.h3_res),
        city=CityConfig(name=cfg.city.name, place=cfg.city.place, crs=cfg.city.crs),
        disruption=DisruptionConfig(
            name=cfg.disruption.name,
            rhos=tuple(float(r) for r in cfg.disruption.rhos),
            n_replicates=int(cfg.disruption.n_replicates),
        ),
        filtration=FiltrationConfig(name=cfg.filtration.name, max_dim=int(cfg.filtration.max_dim)),
        paths=PathsConfig(data=cfg.paths.data, output=cfg.paths.output),
    )

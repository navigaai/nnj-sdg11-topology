import numpy as np
from omegaconf import OmegaConf

from nnj_topology.config import RunConfig, from_omegaconf
from nnj_topology.seeding import set_seed


def test_set_seed_is_reproducible():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_from_omegaconf_builds_frozen_runconfig():
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "h3_res": 8,
            "city": {"name": "testville", "place": "Testville, Country", "crs": "EPSG:3857"},
            "disruption": {"name": "random", "rhos": [0.0, 0.5], "n_replicates": 2},
            "filtration": {"name": "sublevel", "max_dim": 1},
            "paths": {"data": "data", "output": "output"},
        }
    )
    rc = from_omegaconf(cfg)
    assert isinstance(rc, RunConfig)
    assert rc.seed == 7
    assert rc.city.name == "testville"
    assert rc.disruption.rhos == (0.0, 0.5)  # tuple => immutable
    assert rc.h3_res == 8

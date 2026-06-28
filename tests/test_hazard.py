from pathlib import Path

import numpy as np

from nnj_topology.data.hazard import low_elevation_mask

FIX = Path(__file__).parent / "fixtures"


def test_low_elevation_mask_flags_lowest_cells():
    mask, _ = low_elevation_mask(FIX / "mini_dem.tif", quantile=0.2)
    # 20th percentile of [1,2,3,2,5,6,3,6,9] ~ 2.0; cells <= 2.0 are flagged
    assert mask.dtype == bool
    assert mask.sum() >= 1
    assert mask[0, 0]  # elevation 1.0 is lowest -> flagged
    assert not mask[2, 2]  # elevation 9.0 is highest -> not flagged

"""Reproducibility helpers."""
import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["set_seed"]


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Seed set to %d", seed)

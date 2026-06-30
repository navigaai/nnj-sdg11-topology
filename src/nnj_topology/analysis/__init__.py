"""Analysis: district-level inference + city typology overlay."""
from nnj_topology.analysis.crosscity import city_typology
from nnj_topology.analysis.regression import (
    build_district_frame,
    fixed_effects_regression,
    tidy_coefficients,
)

__all__ = [
    "build_district_frame",
    "fixed_effects_regression",
    "tidy_coefficients",
    "city_typology",
]

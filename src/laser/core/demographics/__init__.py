"""Demography utilities for LASER agent populations.

Re-exports:

- [`AliasedDistribution`][laser.core.demographics.AliasedDistribution] —
  Vose alias-method sampler for discrete distributions (e.g., age pyramids).
- [`KaplanMeierEstimator`][laser.core.demographics.KaplanMeierEstimator] —
  predict year-of-death and day-of-death from a cumulative-deaths source.
- [`load_pyramid_csv`][laser.core.demographics.load_pyramid_csv] — loader for
  UN-format CSV age pyramids.

Lower-level utilities such as spatial-population distributions live in
[`spatialpops`][laser.core.demographics.spatialpops] and are imported directly.
"""

from .kmestimator import KaplanMeierEstimator
from .pyramid import AliasedDistribution
from .pyramid import load_pyramid_csv

__all__ = ["AliasedDistribution", "KaplanMeierEstimator", "load_pyramid_csv"]

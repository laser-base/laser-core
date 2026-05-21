"""Benchmark cases for `laser.core.demographics.KaplanMeierEstimator`.

Run with:
    pytest benchmarks/test_kmestimator_benchmarks.py --benchmark-only
"""

import numpy as np

from laser.core.demographics import KaplanMeierEstimator


def _make_cumulative_deaths(max_year=100):
    rng = np.random.default_rng(seed=20260514)
    bumps = rng.integers(0, 1000, size=max_year).astype(np.uint32)
    return np.cumsum(bumps).astype(np.uint32)


def test_predict_year_of_death_n100k(benchmark):
    """Benchmark `predict_year_of_death` for 100,000 individuals."""
    estimator = KaplanMeierEstimator(_make_cumulative_deaths())
    ages = np.random.default_rng(20260514).integers(0, 100, size=100_000).astype(np.uint8)
    benchmark(estimator.predict_year_of_death, ages, max_year=100)


def test_predict_age_at_death_n100k(benchmark):
    """Benchmark `predict_age_at_death` for 100,000 individuals."""
    estimator = KaplanMeierEstimator(_make_cumulative_deaths())
    ages_days = np.random.default_rng(20260514).integers(0, 100 * 365, size=100_000).astype(np.uint32)
    benchmark(estimator.predict_age_at_death, ages_days, max_year=100)

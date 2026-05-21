"""Benchmark cases for `laser.core.distributions`.

Run with:
    pytest benchmarks/test_distributions_benchmarks.py --benchmark-only
"""

import numpy as np

from laser.core import distributions
from laser.core import random as laser_random


def test_constant_n100k(benchmark):
    """Benchmark `distributions.constant` filling 100,000 elements."""
    out = np.empty(100_000, dtype=np.float32)
    benchmark(distributions.constant, 3.0, out)


def test_uniform_n100k(benchmark):
    """Benchmark `distributions.uniform` over 100,000 elements."""
    laser_random.seed(20260514)
    rng = laser_random.prng()
    out = np.empty(100_000, dtype=np.float32)
    benchmark(distributions.uniform, rng, 0.0, 1.0, out)


def test_poisson_n100k(benchmark):
    """Benchmark `distributions.poisson` over 100,000 elements."""
    laser_random.seed(20260514)
    rng = laser_random.prng()
    out = np.empty(100_000, dtype=np.int32)
    benchmark(distributions.poisson, rng, 3.0, out)

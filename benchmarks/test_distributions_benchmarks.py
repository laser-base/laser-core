"""Benchmark cases for `laser.core.distributions`.

These cases use the actual factory-based API: each `distributions.*` factory
returns a Numba-wrapped sampler with signature `(tick, node) -> scalar`, and
`sample_floats` / `sample_ints` fill a pre-allocated buffer in parallel via
`@nb.njit(parallel=True)`.

Run with:
    pytest benchmarks/test_distributions_benchmarks.py --benchmark-only
"""

import numpy as np

from laser.core import distributions as dist
from laser.core import random as laser_random


def test_constant_float_n10m(benchmark):
    """Benchmark `distributions.constant_float` filling 100,000 elements."""
    sampler = dist.constant_float(3.0)
    out = np.empty(10_000_000, dtype=np.float32)
    # Warm up Numba JIT outside the timed call.
    dist.sample_floats(sampler, out)
    benchmark(dist.sample_floats, sampler, out)


def test_uniform_n10m(benchmark):
    """Benchmark `distributions.uniform` over 100,000 elements."""
    laser_random.seed(20260514)
    sampler = dist.uniform(0.0, 1.0)
    out = np.empty(10_000_000, dtype=np.float32)
    dist.sample_floats(sampler, out)  # warmup
    benchmark(dist.sample_floats, sampler, out)


def test_poisson_n10m(benchmark):
    """Benchmark `distributions.poisson` over 10,000,000 elements."""
    laser_random.seed(20260514)
    sampler = dist.poisson(3.0)
    out = np.empty(10_000_000, dtype=np.int32)
    dist.sample_ints(sampler, out)  # warmup
    benchmark(dist.sample_ints, sampler, out)

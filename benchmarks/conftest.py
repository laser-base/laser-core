"""Shared fixtures for the laser-core benchmark suite.

Notes:
    - These benchmarks intentionally use deterministic seeds so that
      perf measurements compare like-for-like across runs.
    - Per-node and per-agent scales are chosen to reflect realistic
      production workloads.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Deterministic NumPy `Generator` shared across all benchmarks."""
    return np.random.default_rng(seed=20260514)


@pytest.fixture
def small_spatial(rng):
    """Spatial inputs at small scale (100 nodes) — quick smoke test."""
    pops = rng.integers(1_000, 100_000, size=100).astype(np.float64)
    coords = rng.uniform(-90, 90, size=(100, 2))
    return pops, coords


@pytest.fixture
def medium_spatial(rng):
    """Spatial inputs at medium scale (1,000 nodes) — realistic provincial workload."""
    pops = rng.integers(1_000, 100_000, size=1_000).astype(np.float64)
    coords = rng.uniform(-90, 90, size=(1_000, 2))
    return pops, coords


def pytest_benchmark_update_machine_info(config, machine_info):
    print()  # put a newline into the output before the benchmark report, so that the machine info is more visible
    print("=" * 80)
    print(f"{machine_info=}")
    print("=" * 80)
    machine_info.clear()
    machine_info.update(
        {
            "ci": "github-actions",
            "runner": "ubuntu-latest",
        }
    )

"""Benchmark cases for the migration models and the `distance` helper.

Failure of these benchmarks (large regressions vs a saved baseline) indicates
that a hot path in `laser.core.migration` has slowed down.

Run with:
    pytest benchmarks/test_migration_benchmarks.py --benchmark-only
"""

import pytest

from laser.core import migration


def _make_distance_matrix(coords):
    return migration.distance(coords[:, 0], coords[:, 1])


def test_distance_n100(benchmark, small_spatial):
    """Benchmark `migration.distance` at small scale (100x100)."""
    _, coords = small_spatial
    benchmark(migration.distance, coords[:, 0], coords[:, 1])


def test_distance_n1000(benchmark, medium_spatial):
    """Benchmark `migration.distance` at medium scale (1,000x1,000)."""
    _, coords = medium_spatial
    benchmark(migration.distance, coords[:, 0], coords[:, 1])


def test_gravity_n100(benchmark, small_spatial):
    """Benchmark `gravity` at small scale."""
    pops, coords = small_spatial
    distances = _make_distance_matrix(coords)
    benchmark(migration.gravity, pops, distances, k=0.01, a=1.0, b=1.0, c=2.0)


def test_gravity_n1000(benchmark, medium_spatial):
    """Benchmark `gravity` at medium scale."""
    pops, coords = medium_spatial
    distances = _make_distance_matrix(coords)
    benchmark(migration.gravity, pops, distances, k=0.01, a=1.0, b=1.0, c=2.0)


def test_competing_destinations_n100(benchmark, small_spatial):
    """Benchmark `competing_destinations` at small scale.

    This call exercises the inner per-pair adjustment loop, which historically
    has been the slowest path among the migration models.
    """
    pops, coords = small_spatial
    distances = _make_distance_matrix(coords)
    benchmark(
        migration.competing_destinations,
        pops,
        distances,
        k=0.01,
        a=1.0,
        b=1.0,
        c=2.0,
        delta=-0.1,
    )


@pytest.mark.parametrize("include_home", [False, True])
def test_stouffer_n100(benchmark, small_spatial, include_home):
    """Benchmark `stouffer` at small scale, both with and without home inclusion."""
    pops, coords = small_spatial
    distances = _make_distance_matrix(coords)
    benchmark(
        migration.stouffer,
        pops,
        distances,
        k=0.01,
        a=1.0,
        b=1.0,
        include_home=include_home,
    )


@pytest.mark.parametrize("include_home", [False, True])
def test_radiation_n100(benchmark, small_spatial, include_home):
    """Benchmark `radiation` at small scale, both with and without home inclusion."""
    pops, coords = small_spatial
    distances = _make_distance_matrix(coords)
    benchmark(
        migration.radiation,
        pops,
        distances,
        k=0.01,
        include_home=include_home,
    )

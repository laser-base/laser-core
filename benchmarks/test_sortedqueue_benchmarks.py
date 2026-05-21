"""Benchmark cases for `laser.core.SortedQueue`.

Run with:
    pytest benchmarks/test_sortedqueue_benchmarks.py --benchmark-only
"""

import numpy as np

from laser.core import SortedQueue


def _make_queue(capacity, n_pre):
    """Helper: build a SortedQueue of `capacity` pre-populated with `n_pre` deterministic entries."""
    sq = SortedQueue(capacity, np.arange(capacity, dtype=np.uint32))
    for i in range(n_pre):
        sq.push(i)
    return sq


def test_sortedqueue_push_10k(benchmark):
    """Benchmark pushing 10,000 entries into an empty SortedQueue of capacity 20,000."""
    capacity = 20_000

    def _run():
        sq = SortedQueue(capacity, np.arange(capacity, dtype=np.uint32))
        for i in range(10_000):
            sq.push(i)

    benchmark(_run)


def test_sortedqueue_pop_10k(benchmark):
    """Benchmark popping 10,000 entries from a pre-populated SortedQueue."""

    def _run():
        sq = _make_queue(capacity=20_000, n_pre=10_000)
        while len(sq) > 0:
            sq.pop()

    benchmark(_run)

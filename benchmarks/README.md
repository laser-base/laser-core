# `benchmarks/` — performance regression tests

`pytest-benchmark` cases for the hot paths of `laser-core`. Kept separate from
`tests/` so they can be opted into explicitly and don't slow down the default
`pytest` invocation.

## Running

```bash
# Requires: uv pip install pytest-benchmark
pytest benchmarks/ --benchmark-only
```

To compare against a saved baseline:

```bash
pytest benchmarks/ --benchmark-only --benchmark-save=main
# ...edit hot code...
pytest benchmarks/ --benchmark-only --benchmark-compare=main
```

## What's covered

| File | Hot path under test |
| --- | --- |
| `test_migration_benchmarks.py` | `gravity`, `competing_destinations`, `stouffer`, `radiation`, `distance` |
| `test_distributions_benchmarks.py` | `distributions.poisson`, `distributions.uniform`, `distributions.constant` |
| `test_kmestimator_benchmarks.py` | `KaplanMeierEstimator.predict_year_of_death`, `predict_age_at_death` |
| `test_sortedqueue_benchmarks.py` | `SortedQueue.push`, `pop`, batch operations |

## Adding a benchmark

When you touch a hot path, add a benchmark covering both the typical and the
largest realistic input size. The threshold values (number of nodes, agents,
samples) should match the largest scale at which the function is expected to
run in production.

```python
def test_my_hot_path_n10000(benchmark):
    """Benchmark `my_hot_path` at production scale (n=10,000)."""
    inputs = make_inputs(n=10_000)
    benchmark(my_hot_path, *inputs)
```

> **TODO**: the baseline thresholds below are starter placeholders. Run the
> suite on the CI hardware and lock in a real baseline before treating these
> numbers as regression gates.

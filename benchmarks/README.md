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

## Updating the baseline

When a perf-impacting change lands and you want to accept the new numbers as the
new floor, trigger the `benchmarks` workflow manually (Actions → benchmarks →
Run workflow → check `save_baseline`), download the resulting artifact, and
replace `.benchmarks/Linux-CPython-3.12-64bit/baseline.json` in a new PR.

### Note

**Python version locks the machine-id**. The path `Linux-CPython-3.12-64bit` is
derived from the runner's Python. If you upgrade the workflow to Python 3.13, you'll
need to recapture and commit a new baseline at `Linux-CPython-3.13-64bit/main.json`.
Either pin Python and call it done, or set up the workflow to capture baselines for
both versions.

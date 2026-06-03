# `benchmarks/` — performance regression tests

`pytest-benchmark` cases for the hot paths of `laser-core`. Kept separate from
`tests/` so they can be opted into explicitly and don't slow down the default
`pytest` invocation.

## Running a single suite

```bash
# Requires: uv pip install pytest-benchmark
pytest benchmarks/ --benchmark-only
```

## Comparing HEAD against `main` (recommended)

The single-suite invocation above gives you absolute numbers but no signal on
"did I regress?". Use `local_compare.py` for that — it spins up a git worktree
at the baseline ref, runs the suite there *and* on HEAD, and compares the two
runs back-to-back on the same workstation (so hardware variance cancels out).

```bash
# Default: compare HEAD against `main`, fail on >25% min-time regression.
.venv/bin/python3 benchmarks/local_compare.py

# Quick smoke (fewer rounds, no failure on regression):
.venv/bin/python3 benchmarks/local_compare.py --rounds=3 --no-fail

# Compare against a tag instead:
.venv/bin/python3 benchmarks/local_compare.py --baseline=v1.0.2

# Tighter gate after you've calibrated to your machine's noise floor:
.venv/bin/python3 benchmarks/local_compare.py --threshold=min:15%

# Leave the worktree in place for debugging:
.venv/bin/python3 benchmarks/local_compare.py --keep-worktree
```

The script writes its outputs to `.benchmarks/local/`:

- `baseline-<ref>.json` — pytest-benchmark JSON from the baseline run
- `head.json` — pytest-benchmark JSON from the current checkout

The temporary worktree lives at `.benchmarks/baseline-worktree/` and is
removed on exit (unless `--keep-worktree` is set).

### Why a local-only tool?

GitHub Actions runners are shared VMs whose performance varies by 30–50%
between runs (noisy neighbours, throttling, cold caches). That noise floor
is well above any per-PR regression we actually want to catch — a useful
threshold like `min:15%` would be drowned out. Running both halves of the
comparison back-to-back on the same workstation cancels out hardware variance
and lets a tight threshold be meaningful. The CI workflow under
`.github/workflows/benchmarks.yml` runs the suite but stays advisory
(`continue-on-error: true`); it is *not* a regression gate.

## Saving / inspecting baselines (single-suite, advanced)

`pytest-benchmark` natively supports save / compare workflows, useful for
ad-hoc inspection without a worktree:

```bash
pytest benchmarks/ --benchmark-only --benchmark-save=main
# ...edit hot code...
pytest benchmarks/ --benchmark-only --benchmark-compare=main
```

These land under `.benchmarks/<machine-id>/`.

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

## When a regression fires

If `local_compare.py` reports a regression and you've confirmed it on a
second run (to rule out noise), you have two options:

1. **Accept the change** — if the regression is intentional (e.g. you traded
   speed for correctness or for an API change), simply re-run after merging.
   The next compare against `main` will use the new code as the baseline.
2. **Investigate** — re-run with `--keep-worktree` to preserve the baseline
   checkout, then use `git diff main..HEAD -- src/laser/core/<area>/` to
   pinpoint the suspect change. The two `.benchmarks/local/*.json` files
   carry per-round timings that `pytest-benchmark compare` (or any JSON
   tool) can drill into.

There is no committed "blessed" baseline file under version control — the
baseline is always whatever `main` resolves to at the moment you run the
script. That intentionally avoids the staleness and machine-pinning problems
of a snapshotted baseline.

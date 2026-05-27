# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 89/100
- **Status**: PASS
- **Date**: 2026-05-26
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 132s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 90/100 | 40% |
| Usability | 83/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **89/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 9/10 | 265 tests on 3-OS / 2-Python CI, bit-equivalence regression tests for every vectorized migration kernel, all API-boundary `assert`s swept to `ValueError`/`TypeError`, `100/tot_pop` magic number removed; one small remaining PRNG-convention slip in `utils.grid()` default population_fn. |
| clear | 9/10 | Modular structure, descriptive names, Google-style docstrings enforced via `interrogate` at 97.9% (95% gate), ruff+black via pre-commit, inline citations on all migration models; only minor gaps: `_cumulative_at_or_closer_2d` lacks a `Raises:` section and the `LaserFrame` class docstring is one sentence. |
| concise | 9/10 | Shared `_validation.py` removes duplication; full vectorization of all four migration models; `lru_cache` prevents redundant Numba JIT; minor cosmetic: `benchmarks.yml` still carries TODO scaffolding inline. |
| simple | 8/10 | Public surface clearly delimited, quick-start runnable, validation errors descriptive; `distributions` workflow still requires a two-step buffer-allocate + `sample_floats` chain. |
| powerful | 8/10 | All migration model parameters keyword-arg-driven, `build_network` accepts any conforming callable, three explicit extension points documented in `architecture.rst`, and three new composition helpers (`mixture2`, `tick_modulated`, `node_modulated`) ship with tests; `sample_floats`/`sample_ints` lack `**kwargs` forwarding for `tick`/`node`. |
| performant | 9/10 | All five migration functions + `distance` fully vectorized (documented in `migration.rst` Performance section); `numba.njit(parallel=True)` with `prange` in distribution kernels; `pytest-benchmark` suite covering migration, distributions, KaplanMeierEstimator, and SortedQueue; CI workflow runs but is still `continue-on-error: true` pending baseline capture. |
| documented | 8/10 | `__init__.py` enumerates the public surface; `CLAUDE.md` + `.claude/skills/laser-core.md` orient AI assistants; `docs/usage.rst` quick-start; `docs/migration.rst` has both Performance characteristics and Choosing a model sections; `docs/architecture.rst` has Extension Points with worked examples; three Sphinx example walkthroughs; `**Example**` blocks on every public distribution. Two gaps: per-folder READMEs not present in `src/laser/core/` or `src/laser/core/demographics/`, and `docs/calibration.rst` doesn't exercise the public API. |
| accessible | 9/10 | PyPI-published (v1.0.2), MIT-licensed, GitHub `laser-base/laser-core`, CHANGELOG/CONTRIBUTING/AUTHORS, ReadTheDocs, README `Support and contact` section, repo-root `CLAUDE.md`, and a Claude Code skill manifest at `.claude/skills/laser-core.md`; the manifest is a static file rather than a live MCP endpoint. |
| compliant | 10/10 | MIT license, no secrets, all runtime deps permissively licensed (BSD/MIT), all key community files present. |
| reproducible | 10/10 | `>=` lower bounds on every runtime dep in `pyproject.toml`; `random.py` exposes a seedable PRNG used throughout; semver tags v0.5.1 → v1.0.2; package published on PyPI (HTTP 200). |

`laser.core` is up to **89/100**, a +3 jump from the prior 86. Quality climbed from 83→90 with the assert-sweep, the magic-number rework, the docstring formula fix, and the `interrogate` gate; Usability climbed from 81→83 with the sampler-composition helpers, the architecture extension docs, and the migration tradeoff/perf sections; Safety stays at a perfect 100. The remaining gaps are tightly scoped: an unenforced benchmark CI gate (needs a CI hardware baseline capture), a couple of per-folder README stubs, the `distributions` two-step sampling ergonomics, and a handful of small docstring polish items.

## Recommendations

1. **performant — Commit the benchmark CI baseline and flip to enforcing** *(effort: medium; automated: partial)*
   This is the single largest remaining lever. Trigger the `benchmarks` workflow via `workflow_dispatch` on a clean main commit (with a `--benchmark-save=main` opt-in input), download the resulting JSON artifact from CI, commit it as `.benchmarks/Linux-CPython-3.12-64bit/main.json`, then in [.github/workflows/benchmarks.yml](.github/workflows/benchmarks.yml) remove `continue-on-error: true` and add a `--benchmark-compare=...main.json --benchmark-compare-fail=min:25%` step. The full step-by-step is documented in the prior conversation. Also update [benchmarks/README.md](benchmarks/README.md) to remove the "starter placeholders" caveat.

2. **simple / powerful — Add a one-shot `sample(...)` convenience to `distributions`** *(effort: quick; automated: yes)*
   Wrap the typical `out = np.empty(N, dtype=...); sample_floats(fn, out)` chain into `distributions.sample(fn, n, *, dtype=None, tick=0, node=0)` that allocates the buffer, dispatches to `sample_floats` or `sample_ints` based on the sampler's return dtype, and returns the array. Keep the existing low-level functions for cases where the caller wants to control buffer allocation.

3. **documented — Add per-folder READMEs to `src/laser/core/` and `src/laser/core/demographics/`** *(effort: quick; automated: yes)*
   Short index pages naming each module's public surface and linking to the relevant `docs/` references. These had been added in an earlier pass and may have been removed; restore them.

4. **correct — Switch `utils.grid()` default `population_fn` to `laser.core.random.prng()`** *(effort: quick; automated: yes)*
   Replace the inline `np.random.uniform(1_000, 100_000)` default in [src/laser/core/utils.py](src/laser/core/utils.py) with a `prng().uniform(...)` call. This is the only place in the project that violates the PRNG-via-`random.py` convention documented in `CLAUDE.md` and is one of the gates that holds `correct` at 9 instead of 10.

5. **clear — Add a `Raises:` section to `_cumulative_at_or_closer_2d` and expand the `LaserFrame` class docstring** *(effort: quick; automated: yes)*
   The private helper documents inputs/outputs but not the `TypeError`/`ValueError` paths inherited from the validation helpers. The `LaserFrame` class docstring is currently one sentence — promote a short overview from the module docstring up to the class.

6. **documented — Rework `docs/calibration.rst` to exercise the public API end-to-end** *(effort: medium; automated: no)*
   The calibration page sits alongside the architecture / migration / usage docs but does not demonstrate a calibration that actually drives the migration models or the demographics utilities. A short worked example (synthetic data + `build_network` + a minimizer wrapper) would close the most concrete documentation gap.

7. **powerful — Forward `**kwargs` through `sample_floats` / `sample_ints` to the sampler** *(effort: quick; automated: yes)*
   Today these helpers only forward `tick` and `node`. Forwarding arbitrary keyword arguments would let composition helpers expose per-call state without baking it into the sampler factory.

8. **accessible — Consider an MCP server in addition to the skill manifest** *(effort: medium; automated: no)*
   The `.claude/skills/laser-core.md` file closed most of the AI-orientation gap. The 10/10 rubric anchor specifically mentions skills *and* MCP; a small MCP server that surfaces the public API and recent `engineering_score.md` would close the remaining usability point. Optional — the skill manifest alone is already meaningful.

9. **concise — Clean up the inline TODO scaffolding in `.github/workflows/benchmarks.yml`** *(effort: quick; automated: yes)*
   Once recommendation #1 lands, delete the two TODO comment blocks. They reference the exact steps that will have been done.

## Full Results

```yaml
project: /Users/christopherlorton/projects/laser-fresh/laser.core
tier: 1
overall_score: 89
failed: false
quality:
  correct:
    score: 9
    weight: 7
    reason: "Comprehensive 265-test suite with per-module coverage including bit-equivalence regression tests for all vectorized migration models and KS-distribution tests for all samplers; CI matrix runs Python 3.10 and 3.14 on three OSes; migration models cite published references; API boundaries use `TypeError`/`ValueError`. The one remaining PRNG-convention slip is in `utils.grid()` whose default `population_fn` uses `np.random.uniform` directly instead of `prng()`."
  clear:
    score: 9
    weight: 2
    reason: "One-module-per-concern layout with descriptive names, Google-style docstrings enforced via interrogate at 97.9% coverage (95% gate), ruff/black pre-commit hooks, and inline citations for all published model parameters. The only gaps are that `_cumulative_at_or_closer_2d` lacks a `Raises:` section and the `LaserFrame` class docstring is a single sentence."
  concise:
    score: 9
    weight: 1
    reason: "No avoidable duplication; shared validation extracted to `_validation.py`; vectorized NumPy replaces O(N^2) Python loops in all migration models; `lru_cache` prevents redundant Numba JIT compilation; ruff+black enforced via pre-commit. The only marginal issue is the inline TODO scaffolding in `benchmarks.yml`."
usability:
  simple:
    score: 8
    weight: 3
    reason: "Public surface clearly delimited in `__init__.py` and `docs/usage.rst` provides a runnable quick-start; `_validation` raises descriptive `ValueError`/`TypeError`. The gap holding it back from 9 is the two-step buffer-allocate + `sample_floats` distribution-sampling chain, which could be a one-liner with a convenience wrapper."
  powerful:
    score: 8
    weight: 2
    reason: "All migration model parameters are keyword arguments; `build_network` accepts any callable following the `(pops, distances, **params)` convention; three explicit extension points documented in `docs/architecture.rst`; new composition helpers `mixture2`, `tick_modulated`, `node_modulated` expand the distributions API. The slight gap is that `sample_floats`/`sample_ints` lack `**kwargs` forwarding to the sampler."
  performant:
    score: 9
    weight: 2
    reason: "All four migration models plus `distance` fully vectorized (documented in `docs/migration.rst` Performance characteristics); `distributions.sample_floats`/`sample_ints` use `numba.njit(parallel=True)` with `prange`; `pytest-benchmark` suite covering migration, distributions, `KaplanMeierEstimator`, and `SortedQueue` lives under `benchmarks/`. The CI workflow runs but is still `continue-on-error: true` with no committed baseline."
  documented:
    score: 8
    weight: 2
    reason: "Multi-layered docs: `__init__.py`, `CLAUDE.md`, `.claude/skills/laser-core.md`, `docs/usage.rst` quick-start, `docs/migration.rst` (Choosing a model + Performance characteristics), `docs/architecture.rst` (Extension Points), three Sphinx walkthroughs, `**Example**` blocks on every public distribution. Two gaps: per-folder READMEs not present in `src/laser/core/` or `src/laser/core/demographics/`, and `docs/calibration.rst` doesn't exercise the public API."
  accessible:
    score: 9
    weight: 1
    reason: "Published on PyPI as `laser-core` v1.0.2 (single-command install), MIT license, GitHub `laser-base/laser-core`, ships `CHANGELOG.rst`, `CONTRIBUTING.rst`, `AUTHORS.rst`, `pyproject.toml`-based build, README `Support and contact` section, repo-root `CLAUDE.md`, and a Claude Code skill manifest at `.claude/skills/laser-core.md`. The skill manifest is a static file rather than a registered MCP endpoint."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license confirmed at the repo root; no exposed secrets, API keys, or PII found; all key runtime dependencies (numpy, numba, pandas, geopandas, shapely, h5py, matplotlib, click) carry BSD/MIT/permissive licenses; AUTHORS.rst, CHANGELOG.rst, and CONTRIBUTING.rst all present."
  reproducible:
    score: 10
    weight: 4
    reason: "All key dependencies have `>=` lower-bound pins in `pyproject.toml`; `laser.core.random` exposes a fully seedable PRNG used throughout; semver git tags v0.5.1 → v1.0.2; PyPI returned HTTP 200 confirming the package is published."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.
- **Score delta vs. previous audit (86 → 89, +3)**:
  - **Quality 83→90 (+7)**: `correct` 8→9 (asserts swept, magic number removed, formulas fixed) and `concise` held at 9 (vectorization complete) and `clear` held at 9 (interrogate gate active).
  - **Usability 81→83 (+2)**: `performant` 8→9 (stouffer/radiation now vectorized, benchmark CI live as advisory) and `simple`/`powerful`/`documented`/`accessible` held at their prior 8/8/8/9 values.
  - **Safety 100 (held)**: no changes; both metrics already maxed.
- **Top remaining lever**: locking in benchmark baselines on CI hardware and flipping the workflow to enforcing (`continue-on-error: false` + `--benchmark-compare-fail`). That alone would project `performant` 9→10 and overall 89→~90.

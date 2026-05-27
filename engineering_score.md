# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 89/100
- **Status**: PASS
- **Date**: 2026-05-27
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 167s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 90/100 | 40% |
| Usability | 83/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **89/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 9/10 | 271 tests on 3-OS / 2-Python CI, given/when/then test docstrings, bit-equivalence regression coverage on every vectorized migration kernel, all API-boundary `assert`s swept, magic number removed, formulas fixed; remaining nits: `LaserFrame.add` docstring still says `AssertionError` while the code now raises `ValueError`, and the `RE = 6371.0` constant is documented inline but not formally cited. |
| clear | 9/10 | One-module-per-concern layout, `interrogate` at 97.8% (95% gate), Google-style docstrings on all public APIs; `SortedQueue` class docstring is sparse (no Attributes/Example) compared to the rest of the codebase. |
| concise | 9/10 | Validation deduplicated into `_validation.py`, migration kernels fully vectorized, `lru_cache` on distribution factories, ruff+black via pre-commit; only a small parameter-validation repetition pattern in `_sanity_checks`. |
| simple | 8/10 | Public surface clearly enumerated, `build_network` and the new `sample()` collapse common workflows into one-liners, validation errors descriptive; **concrete bug**: the distribution snippet in `docs/usage.rst` uses a stale API (`distributions.poisson(rng, lam=3.0, out=out)`) that doesn't match the actual factory + `sample_floats`/`sample_ints` (or new `sample()`) pattern. |
| powerful | 9/10 | Four migration models with all parameters exposed; three sampler-composition helpers (`mixture2`, `tick_modulated`, `node_modulated`); `build_network` accepts any conforming callable; `LaserFrame` subclassing pattern documented with worked examples in `docs/architecture.rst`; `PropertySet` composes with `+=`/`<<=`/`|=`. No plugin registry, but the plain-function convention is documented. |
| performant | 7/10 | All five migration models and `distance` fully vectorized; `numba.njit(parallel=True)` on hot kernels; `pytest-benchmark` suite covering migration / distributions / KMEstimator / SortedQueue. **Two concrete gaps**: (a) CI workflow still runs `continue-on-error: true` with no committed baseline, so regressions are not actually gated; (b) `benchmarks/test_distributions_benchmarks.py` references a stale API (`distributions.constant`, `distributions.poisson(rng, ...)`) that would fail on execution. |
| documented | 9/10 | Multi-layered docs: `__init__.py`, `CLAUDE.md`, `.claude/skills/laser-core.md`, `docs/usage.rst` quick-start, `docs/migration.rst` (Choosing a model + Performance), `docs/architecture.rst` (Extension Points), three Sphinx walkthroughs, `**Example**` blocks on every public distribution. Same stale `usage.rst` distribution snippet flagged under `simple` also affects this metric; per-folder READMEs under `src/laser/core/` / `src/laser/core/demographics/` not present. |
| accessible | 9/10 | PyPI-published (v1.0.2), MIT-licensed, `CHANGELOG.rst` / `CONTRIBUTING.rst` / `AUTHORS.rst`, README `Support and contact` section, repo-root `CLAUDE.md`, Claude Code skill manifest at `.claude/skills/laser-core.md`. No `CODE_OF_CONDUCT.md`. |
| compliant | 10/10 | MIT license, no secrets, all runtime deps permissively licensed (BSD/MIT), all key community files present. |
| reproducible | 10/10 | `>=` lower bounds on every runtime dep, `uv.lock` committed, deterministic seeding via `random.py` used throughout (including the recently-fixed `utils.grid` default `population_fn`), semver tags through v1.0.2, package on PyPI (HTTP 200). |

`laser.core` holds at **89/100** overall, with meaningful internal movement: `powerful` 8→9 (sampler-composition helpers + `sample()` + `build_network` all credited), `documented` 8→9 (LaserFrame class docstring + Extension Points + Choosing a model sections), and `performant` 8→7 (regression — the auditor surfaced two concrete bugs: the still-advisory benchmark CI gate AND a stale distributions-API snippet in both `docs/usage.rst` and `benchmarks/test_distributions_benchmarks.py`). Safety stays perfect at 100/100. The two stale-API bugs are the most actionable next items; fixing them alone would project ~92/100.

## Recommendations

1. **performant / simple / documented — Fix the stale `distributions` API references** *(effort: quick; automated: yes)*
   Two specific places call a `distributions.poisson(rng, lam=3.0, out=out)` / `distributions.constant(...)` API that does not exist; the actual pattern is `sampler = distributions.poisson(lam=3.0)` followed by `distributions.sample_floats(sampler, out)` (or the new one-liner `distributions.sample(distributions.poisson, n=N, lam=3.0)`). Update [docs/usage.rst](docs/usage.rst) (the "Sampling from a distribution" snippet) and [benchmarks/test_distributions_benchmarks.py](benchmarks/test_distributions_benchmarks.py) (every benchmark function). This single fix recovers `performant` 7→8 (the runtime bug), `simple` 8→9, and `documented` 9→10.

2. **performant — Commit the benchmark CI baseline and flip the workflow to enforcing** *(effort: medium; automated: partial)*
   The largest remaining lever for `performant`. Trigger the `benchmarks` workflow via `workflow_dispatch` on a clean main commit with a `--benchmark-save=main` opt-in, download the JSON artifact, commit it as `.benchmarks/Linux-CPython-3.12-64bit/main.json`, then remove `continue-on-error: true` from [.github/workflows/benchmarks.yml](.github/workflows/benchmarks.yml) and add the comparison step with `--benchmark-compare-fail=min:25%`. Step-by-step is documented in a prior conversation.

3. **correct — Fix the docstring on `LaserFrame.add` (and any sibling) that still mentions `AssertionError`** *(effort: quick; automated: yes)*
   The implementation was updated to raise `ValueError` but the docstring still advertises `AssertionError`. Update the `Raises:` section to match.

4. **clear — Flesh out the `SortedQueue` class docstring** *(effort: quick; automated: no)*
   Bring it in line with the recently-expanded `LaserFrame` and `AliasedDistribution` class docstrings: short overview, `Attributes:` block, and a runnable `**Example**` showing push/peek/pop.

5. **documented — Add per-folder READMEs under `src/laser/core/` and `src/laser/core/demographics/`** *(effort: quick; automated: yes)*
   These had been added in an earlier pass; restore them so each major area has a one-page index of its public surface and pointers to the relevant `docs/` references.

6. **accessible — Add a `CODE_OF_CONDUCT.md`** *(effort: quick; automated: yes)*
   Use the standard Contributor Covenant 2.1 template; reference it from `CONTRIBUTING.rst`.

7. **correct — Add a citation comment for `RE = 6371.0`** *(effort: quick; automated: no)*
   One-liner: cite IUGG / WGS84 / a standard reference next to the existing `# Earth radius in km` comment in [src/laser/core/migration.py](src/laser/core/migration.py).

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
    reason: "271 tests pass on 3-OS / 2-Python CI with ruff+black+interrogate enforced via pre-commit, tests carry given/when/then docstrings and serve as documentation. Migration models cite published references; the single remaining undocumented magic number is `RE = 6371.0` (with an inline `# Earth radius in km` comment but no formal IUGG/WGS84 citation), and `LaserFrame.add` docstring still references `AssertionError` while the implementation now raises `ValueError`."
  clear:
    score: 9
    weight: 2
    reason: "Well-organized one-module-per-concern structure; interrogate reports 97.8% docstring coverage (gate is 95%); all public API functions have Google-style docstrings with Args/Returns/Raises/Example blocks. `SortedQueue` class docstring is sparse (no Attributes or Example block) compared to the rest of the codebase."
  concise:
    score: 9
    weight: 1
    reason: "No significant copy-paste duplication; vectorized NumPy and Numba parallel kernels replace per-agent/per-row Python loops throughout; `_sanity_checks` centralizes repetitive input validation. The validation block uses a repeated `params.get()` two-liner per parameter that could be a small loop, but this is a minor style nit."
usability:
  simple:
    score: 8
    weight: 3
    reason: "Public surface clearly defined; `build_network` and `sample` collapse common workflows into one-liners; thorough `ValueError`/`TypeError` validation at all boundaries. Concrete bug: the distribution snippet in `docs/usage.rst` shows a stale `distributions.poisson(rng, lam=3.0, out=out)` API that does not match the actual factory pattern, which would mislead a new user copying the quick-start."
  powerful:
    score: 9
    weight: 2
    reason: "Four migration models with fully exposed parameters; three sampler-composition helpers (mixture2, tick_modulated, node_modulated); `build_network` accepts any compliant callable; `LaserFrame` subclassable with worked examples; `PropertySet` supports +=/<<=/|= composition. No plugin registry but the plain-function convention is explicitly the preferred extension path."
  performant:
    score: 7
    weight: 2
    reason: "All five migration models and `distance` fully vectorized; sampling hot paths use `numba.njit(parallel=True)`; pytest-benchmark suite covers migration, KM estimator, and SortedQueue. The CI benchmark workflow runs `continue-on-error: true` with no committed baseline so no regression is actually caught, and `benchmarks/test_distributions_benchmarks.py` calls a stale API (`distributions.constant`, `distributions.poisson(rng, ...)`) that does not exist in the current module and would fail on execution."
  documented:
    score: 9
    weight: 2
    reason: "Comprehensive docs covering quick-start, design principles + subclassing patterns, migration model tradeoffs and numerical edge cases, performance guidance, and three end-to-end tutorials plus Jupyter notebooks; all 13 distribution factories have runnable `**Example**` blocks; `LaserFrame` class docstring covers the full design. The stale distribution example in `usage.rst` and the absence of per-folder READMEs under `src/laser/core/` and `src/laser/core/demographics/` keep this just below a perfect score."
  accessible:
    score: 9
    weight: 1
    reason: "Published on PyPI (`pip install laser-core`), MIT license, CHANGELOG.rst, CONTRIBUTING.rst, maintainer emails and issue tracker in `README.rst`, and AI-orientation files (`CLAUDE.md` + `.claude/skills/laser-core.md`). The only standard file missing is a `CODE_OF_CONDUCT.md`, which is expected for a Tier-1 community-facing library."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license confirmed at the repo root; no hardcoded secrets, tokens, or PII found anywhere in source; all runtime dependencies (numpy, numba, pandas, geopandas, shapely, h5py, matplotlib, click) carry BSD/MIT/permissive licenses; AUTHORS.rst, CONTRIBUTING.rst, and CHANGELOG.rst with an Unreleased section are all present."
  reproducible:
    score: 10
    weight: 4
    reason: "All runtime deps in `pyproject.toml` carry `>=` lower-bound pins; `uv.lock` provides a full lock file; semantic version tags run through v1.0.2; `laser-core` is published on PyPI (HTTP 200); process-wide PRNG seedability is enforced via `laser.core.random.seed()` and now consumed by every in-package call site including the recently-fixed `utils.grid` default `population_fn`."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.
- **Score delta vs. previous audit (89 → 89, flat overall, with internal churn)**:
  - **Improved**: `powerful` 8→9 (sampler-composition helpers + `sample()` + `build_network` credited), `documented` 8→9 (LaserFrame class docstring + Choosing a model + Extension Points sections).
  - **Regressed**: `performant` 8→7 — the auditor surfaced two concrete bugs that had been latent: the benchmark CI gate is still advisory (known) AND `benchmarks/test_distributions_benchmarks.py` references a stale `distributions.constant` / `distributions.poisson(rng, ...)` API that would fail to execute.
  - **New observations not previously flagged**: stale distribution snippet in `docs/usage.rst`, `LaserFrame.add` docstring still says `AssertionError`, `SortedQueue` class docstring is sparse, missing `CODE_OF_CONDUCT.md`.
- **Top remaining lever**: fixing recommendation #1 (the stale `distributions` API references) is a quick, automatable change that would clear the new `performant`-7 regression AND raise both `simple` and `documented` to 9/10 simultaneously — projecting overall **89 → ~92**.

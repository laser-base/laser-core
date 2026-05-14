# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 84/100
- **Status**: PASS
- **Date**: 2026-05-13
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 136s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 87/100 | 40% |
| Usability | 76/100 | 40% |
| Safety | 96/100 | 20% |
| **Total** | **84/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 9/10 | Comprehensive multi-platform CI/CD, 225 tests, models cite published references; minor `assert`-for-validation and one unexplained magic number. |
| clear | 8/10 | Modular package with Google-style docstrings on most APIs; a handful of public properties and one `__init__` lack proper docstrings. |
| concise | 8/10 | Input-validation helpers duplicated between [src/laser/core/migration.py](src/laser/core/migration.py) and [src/laser/core/laserframe.py](src/laser/core/laserframe.py); otherwise tight, NumPy/Numba-idiomatic code. |
| simple | 7/10 | Clean exported classes with validation and clear error messages, but [docs/usage.rst](docs/usage.rst) is essentially empty and submodules (`migration`, `distributions`, `random`, `demographics`) are not surfaced at the top level. |
| powerful | 8/10 | Migration parameters, RNG seeds, and distributions are all composable; subclassing `LaserFrame` is not explicitly demonstrated. |
| performant | 7/10 | Hot paths use Numba `parallel=True` and vectorized NumPy, but `competing_destinations` has an O(n²) Python double loop and `distance()` iterates in Python over `lat1.size`. No formal benchmark suite. |
| documented | 8/10 | Strong docs (architecture, three worked examples, migration & KMEstimator references, performance guide, calibration); gaps include empty `usage.rst`, no per-folder READMEs, and submodules not advertised via `__init__.__all__`. |
| accessible | 9/10 | Published on PyPI as `laser-core` (15+ releases), MIT-licensed, public GitHub repo, ReadTheDocs hosted, all community files present; no `CLAUDE.md`/skills/MCP markers for AI-optimization. |
| compliant | 10/10 | MIT license, no hardcoded secrets, all runtime deps are permissively licensed (BSD/MIT), and all key community files (CHANGELOG, CONTRIBUTING, AUTHORS) are present. |
| reproducible | 9/10 | PyPI-published with semver git tags (v0.0.1 → v1.0.1), `uv.lock` present, deterministic seeding via [src/laser/core/random.py](src/laser/core/random.py); however core deps in [pyproject.toml](pyproject.toml) lack `>=` version pins. |

`laser.core` is a strong Tier 1 library: high-quality test coverage on multi-platform CI, MIT license, PyPI publication with semantic versioning, deterministic seeding, and comprehensive documentation with worked examples. The biggest gaps are in usability polish — a near-empty `usage.rst`, submodules that aren't surfaced from the package root, a couple of non-vectorized hot loops in migration code, and missing docstrings on a few public properties — plus minor reproducibility (no `>=` version bounds on dependencies) and conciseness (duplicated validation helpers) deductions. No failure conditions triggered.

## Recommendations

1. **simple — Flesh out `docs/usage.rst` and surface submodules at the package root** *(effort: medium; automated: no)*
   Replace the one-line [docs/usage.rst](docs/usage.rst) with a quick-start that imports the package, instantiates `LaserFrame`/`PropertySet`/`SortedQueue`, calls into `migration` and `distributions`, and seeds the RNG via `random.seed()`. Update [src/laser/core/__init__.py](src/laser/core/__init__.py) so `migration`, `distributions`, `random`, and `demographics` are re-exported (or at minimum listed in `__all__`) so users — and IDEs — discover the public surface without spelunking the repo.

2. **performant — Vectorize the two remaining Python hot loops in `migration.py`** *(effort: medium; automated: no)*
   In [src/laser/core/migration.py](src/laser/core/migration.py), rewrite `competing_destinations` (the O(n²) double `for` loop near lines 197–200) using outer-product broadcasting over the distance matrix, and replace the per-row Python loop in `distance()` with a fully vectorized haversine over `lat1`/`lon1`/`lat2`/`lon2` arrays. Add a small `pytest-benchmark`-based regression test under `tests/` so future regressions are caught automatically.

3. **documented — Add a benchmark/performance harness and per-folder READMEs** *(effort: medium; automated: no)*
   Introduce a `benchmarks/` directory (or `tests/benchmarks/`) with `pytest-benchmark` cases for migration models, distributions, `KaplanMeierEstimator`, and `SortedQueue`. Add short `README.md` files under [src/laser/core/](src/laser/core/), [src/laser/core/demographics/](src/laser/core/demographics/), and [examples/](examples/) that point readers to the relevant tutorial pages and document the intended entry points.

4. **clear — Fill in missing docstrings on public APIs** *(effort: quick; automated: partial)*
   Convert the inline string in `AliasedDistribution.__init__` into a proper Google-style docstring, and add docstrings to the `alias`/`probs`/`total` properties in `pyramid.py`, `_pdod` in `kmestimator.py`, and `sum_populations_as_close_or_closer` in `migration.py`. A pre-commit hook such as `interrogate` or `pydocstyle` configured at >=95% coverage will prevent regressions.

5. **concise — Extract the duplicated validation helpers into a shared module** *(effort: quick; automated: no)*
   The helpers `_is_instance`, `_is_dtype`, `_has_shape`, `_has_dimensions`, and `_has_values` are duplicated between [src/laser/core/migration.py](src/laser/core/migration.py) and [src/laser/core/laserframe.py](src/laser/core/laserframe.py). Move them into a new internal module (e.g. `src/laser/core/_validation.py`) and import from there in both call sites.

6. **correct — Replace `assert` with `ValueError` and document the `100 / tot_pop` magic number** *(effort: quick; automated: no)*
   In `calc_capacity` (and any other API-facing validation site that currently uses bare `assert`), raise `ValueError`/`TypeError` so the check survives `python -O`. In [src/laser/core/demographics/spatialpops.py](src/laser/core/demographics/spatialpops.py) line ~60, add a comment (and ideally a named constant or function argument) explaining why `nsizes = np.minimum(nsizes, 100 / tot_pop)` — what does the `100` represent demographically? — or cite the source.

7. **reproducible — Add lower-bound version pins on core dependencies** *(effort: quick; automated: no)*
   In [pyproject.toml](pyproject.toml), add `>=` constraints on `numpy`, `numba`, `pandas`, `matplotlib`, `geopandas`, `shapely`, and `h5py` reflecting the minimum versions currently exercised on CI (matching what `uv.lock` shows is being used). Keep upper bounds off to avoid resolver pain unless a known incompatibility forces one.

8. **accessible — Add AI-optimization affordances** *(effort: quick; automated: no)*
   Add a `CLAUDE.md` at the repo root summarizing project conventions, how to run tests and the docs build, and the canonical public surface. Optionally publish a small skills bundle (or note where to find one) so AI assistants can be steered toward the right entry points and idioms. This is the only remaining Tier 1 accessibility gap.

## Full Results

```yaml
project: /Users/christopherlorton/projects/laser-fresh/laser.core
tier: 1
overall_score: 84
failed: false
quality:
  correct:
    score: 9
    weight: 7
    reason: "Comprehensive CI/CD tests Python 3.9–3.14 on three platforms with 225 test functions covering all major modules, including mathematical validation of migration models against reference implementations with published citations, KS tests for statistical distributions, overflow tests, and edge cases; tests double as documentation. Minor deductions: `calc_capacity` uses suppressible `assert` for input validation instead of `ValueError`, and `spatialpops.py` has an unexplained magic number `100 / tot_pop`."
  clear:
    score: 8
    weight: 2
    reason: "Well-organized modular structure with comprehensive Google-style docstrings on nearly all public APIs, consistent naming, and ruff+black style enforcement; however `AliasedDistribution.__init__` has an inline string rather than a proper docstring, the `alias`/`probs`/`total` properties in `pyramid.py` have no docstrings, `_pdod` in `kmestimator.py` has no docstring, and `sum_populations_as_close_or_closer` in `migration.py` has no docstring."
  concise:
    score: 8
    weight: 1
    reason: "The five input-validation helper functions (`_is_instance`, `_is_dtype`, `_has_shape`, `_has_dimensions`, `_has_values`) are duplicated between `migration.py` and `laserframe.py` rather than shared from a common module; otherwise NumPy and Numba are used appropriately throughout, and ruff+black enforcement keeps style consistent."
usability:
  simple:
    score: 7
    weight: 3
    reason: "Three clearly exported public classes (`LaserFrame`, `PropertySet`, `SortedQueue`) with sensible defaults and thorough `ValueError`/`TypeError` messages; a full SIR demo runs in ~40 lines. `docs/usage.rst` is nearly empty (one line) and submodule APIs (`distributions`, `migration`, `random`, `demographics`) are not surfaced in `__init__.py` or a quick-start, requiring users to discover them independently."
  powerful:
    score: 8
    weight: 2
    reason: "Random seed is fully settable via `laser.core.random.seed()`; all migration model parameters (k, a, b, c, delta) are exposed; `LaserFrame` supports scalar/vector/array properties; `PropertySet` offers five distinct merge operators; Numba distributions use `lru_cache` closures for composable parameterization. Subclassing `LaserFrame` is not explicitly documented or demonstrated."
  performant:
    score: 7
    weight: 2
    reason: "`KaplanMeierEstimator`, `SortedQueue`, and distributions use Numba njit with `parallel=True` for large arrays; the gravity model is fully vectorized. However, `competing_destinations` has an O(n^2) double Python for-loop (migration.py lines ~197–200) and `distance()` loops over `lat1.size` in Python rather than using outer-product broadcasting; informal timing tests exist but no formal benchmark suite or performance regression tests."
  documented:
    score: 8
    weight: 2
    reason: "Documentation covers architecture, three full worked examples (SIR, VitalDynamics, spatial), migration model reference, performance guide, calibration, and `KaplanMeierEstimator` reference; major classes have docstrings with code examples. Gaps: `docs/usage.rst` has only one line, no per-folder READMEs, and `distributions`/`migration`/`random`/`demographics` are not listed in `__init__.__all__` or highlighted as top-level UIs."
  accessible:
    score: 9
    weight: 1
    reason: "Published on PyPI (`pip install laser-core`, 15+ versions), MIT license, GitHub public repo at `laser-base/laser-core` with `CHANGELOG.rst`, `CONTRIBUTING.rst`, `AUTHORS.rst`, ReadTheDocs badge, and maintainer contact emails in `pyproject.toml`. No `CLAUDE.md`, MCP server definition, or skills file present, so AI-optimization markers for Tier 1 are absent."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license present at the repository root; no hardcoded secrets found; all runtime dependencies (numpy, numba, pandas, geopandas, etc.) are permissively licensed (BSD/MIT); all key community files (`CHANGELOG.rst`, `CONTRIBUTING.rst`, `AUTHORS.rst`) are present."
  reproducible:
    score: 9
    weight: 4
    reason: "Package is published on PyPI as `laser-core`, has full semantic versioning with git tags (v0.0.1 through v1.0.1), a `uv.lock` file, and a dedicated `random.py` module with configurable seeds; however key runtime dependencies (numpy, numba, matplotlib, etc.) have no version bounds at all in `pyproject.toml`, which would need at least `>=` pins to reach a score of 10."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.

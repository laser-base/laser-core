# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 86/100
- **Status**: PASS
- **Date**: 2026-05-21
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 139s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 83/100 | 40% |
| Usability | 81/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **86/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 8/10 | 228 tests on 3-OS / 2-Python CI, plus new vectorization regression tests; remaining `assert` usages in propertyset.py and utils.py and the unexplained magic number in spatialpops.py keep this below 10. |
| clear | 9/10 | Google-style docstrings on all public APIs, shared `_validation.py` keeps guard logic in one place; class-level docstring on `AliasedDistribution.__init__` and parameter docs on a couple of `LaserFrame` helpers are the remaining gaps. |
| concise | 9/10 | Dedup'd validation helpers and vectorized hot paths; minor remaining duplication in parametrizable test pairs. |
| simple | 8/10 | Clean public surface signposted in `__init__.py`, CLAUDE.md, per-folder READMEs, and a fleshed-out `docs/usage.rst`; remaining issue is that `stouffer`/`radiation` still loop over source nodes in Python. |
| powerful | 8/10 | All migration model parameters exposed, `LaserFrame`/`PropertySet` highly composable, Numba distributions use `lru_cache` closures; no documented subclassing/plugin hook for custom migration models. |
| performant | 8/10 | `competing_destinations` and `distance` are now vectorized; Numba `parallel=True` on hot kernels; benchmark suite is scaffolded but baselines aren't locked into CI; `stouffer` and `radiation` still loop over source nodes. |
| documented | 8/10 | Architecture, three worked examples, migration & KMEstimator references, performance guide, calibration, quick-start, per-folder READMEs, CLAUDE.md; gaps: no runnable examples in `distributions` docstrings, open TODOs in `docs/migration.rst`, no parameter-choice tradeoff guide. |
| accessible | 9/10 | PyPI-published (`laser-core`, 15+ versions), MIT license, ReadTheDocs, CHANGELOG/CONTRIBUTING/AUTHORS, CLAUDE.md for AI assistants; no MCP/skills manifest and the maintainer contact isn't surfaced in the README. |
| compliant | 10/10 | MIT license, no secrets, all deps permissively licensed, all community files present. |
| reproducible | 10/10 | `>=` lower bounds on key deps in `pyproject.toml`, `uv.lock` for full pinning, deterministic seeding via `random.py`, semver tags through v1.0.2 on PyPI. |

`laser.core` improved from 84/100 to 86/100 since the last audit. Safety is now perfect at 100/100 after adding `>=` version bounds. Quality and usability both improved meaningfully (clarity, conciseness, simplicity, performance, and reproducibility all gained a point), but the quality.correct score actually dropped 9→8 because the latest review surfaced additional `assert` usages in `propertyset.py` and `utils.py` that hadn't been flagged before; the original `100 / tot_pop` magic number in `spatialpops.py` also remains unexplained. The biggest remaining usability gaps are the Python-level loops in `stouffer`/`radiation` and the absence of runnable examples in the `distributions` docstrings.

## Recommendations

1. **correct — Replace remaining `assert` statements at API boundaries with `ValueError`/`TypeError`** *(effort: quick; automated: yes)*
   Sweep `assert` calls intended for input validation across [src/laser/core/propertyset.py:201,243,283](src/laser/core/propertyset.py) and [src/laser/core/utils.py:193](src/laser/core/utils.py) (`initialize_population`). Replace each with a proper exception so the check survives `python -O`. Internal-consistency asserts (e.g. those in `predict_year_of_death` and `predict_age_at_death`) can stay because they are not user-facing.

2. **correct — Explain or replace the `100 / tot_pop` magic number** *(effort: quick; automated: no)*
   In [src/laser/core/demographics/spatialpops.py](src/laser/core/demographics/spatialpops.py) (around line 60), either add a comment citing the source of `100` (what it represents demographically — a max-share fraction? a per-mille cap?) or lift it to a named module-level constant / function parameter with a default. Requires domain knowledge.

3. **performant — Vectorize the source-node loop in `stouffer` and `radiation`** *(effort: medium; automated: no)*
   In [src/laser/core/migration.py](src/laser/core/migration.py), both `stouffer` and `radiation` currently iterate `for i in range(len(pops))` to call `sum_populations_as_close_or_closer` per source node. Lift this into a batched/vectorized version (e.g. sort per-row, build cumulative sums in a single 2D operation, handle equidistant ties with a single grouped maximum). Re-use the bit-equivalence regression-test pattern in [tests/test_migration.py](tests/test_migration.py) to guarantee no change in numerical output.

4. **documented — Add runnable examples to the `distributions` docstrings** *(effort: quick; automated: partial)*
   For each public function in [src/laser/core/distributions.py](src/laser/core/distributions.py), add a `**Example**` block that shows the typical pattern (acquire `prng()`, pre-allocate the output buffer, call the function). The mathematical formula is already documented; adding the wire-up example is what gets users from "I read the docs" to "I'm running it."

5. **documented — Close the open TODOs in `docs/migration.rst` and add a parameter-tradeoff guide** *(effort: medium; automated: no)*
   Address the input-check / precision TODOs flagged inline in [docs/migration.rst](docs/migration.rst) and add a short guide comparing the four migration models (gravity, competing-destinations, Stouffer, radiation) — when to pick each, what the parameter ranges mean, and what numerical edge cases to watch out for.

6. **performant — Lock benchmark baselines into CI as regression gates** *(effort: medium; automated: partial)*
   Run the [benchmarks/](benchmarks/) suite on the CI hardware to collect a saved baseline, then add a GitHub Actions step that runs `pytest benchmarks/ --benchmark-only --benchmark-compare=<baseline>` and fails the build on a configurable regression threshold. Update [benchmarks/README.md](benchmarks/README.md) to remove the "starter placeholders" caveat once real baselines are in place.

7. **simple — Surface the now-vectorized `competing_destinations` performance characteristics** *(effort: quick; automated: no)*
   Update [docs/migration.rst](docs/migration.rst) and any benchmark numbers to reflect the now-vectorized implementation, and consider adding a "Performance" subsection to each migration model's reference so users know the relative cost of choosing one over another.

8. **clear — Add the missing docstrings on `AliasedDistribution.__init__` class-level and `LaserFrame` `_save` helpers** *(effort: quick; automated: yes)*
   Fill in the few remaining gaps. Adding a `pre-commit` hook like `interrogate` with `--fail-under=95` would prevent regressions here.

9. **powerful — Document the supported subclassing / extension points** *(effort: medium; automated: no)*
   In [docs/architecture.rst](docs/architecture.rst), add a short section showing how to subclass `LaserFrame`, extend `PropertySet`, or register a custom migration model. Even a single end-to-end example is enough to unlock the "powerful" rubric.

10. **accessible — Add an MCP / skills manifest for the LASER public API** *(effort: medium; automated: no)*
    Publish a small MCP server or `.claude/skills/` bundle (the IDM marketplace already hosts companion skill plugins) that AI assistants can attach to direct users toward the canonical idioms documented in `CLAUDE.md`. Also surface the maintainer contact in the README so non-AI users can find it without digging into `pyproject.toml`.

## Full Results

```yaml
project: /Users/christopherlorton/projects/laser-fresh/laser.core
tier: 1
overall_score: 86
failed: false
quality:
  correct:
    score: 8
    weight: 7
    reason: "Comprehensive tests (228 functions across 11 files) cover main workflows, edge cases, overflow, and include bit-equivalence regression tests for recently vectorized code; CI runs on Python 3.10 and 3.14 across Linux/macOS/Windows via GitHub Actions with tox. The unexplained `100 / tot_pop` magic number in `spatialpops.py` and `assert` statements still present in `propertyset.py` (lines 201, 243, 283) and `utils.py` line 193 (`initialize_population`) are bypassed under `python -O`, leaving misuse silently unreported in optimized mode."
  clear:
    score: 9
    weight: 2
    reason: "Well-organized modular structure with descriptive names; all public APIs have Google-style docstrings including mathematical formulas, parameter descriptions, raises clauses, and examples; `_validation.py` cleanly centralizes guard logic; minor remaining gap is the class-level docstring on `AliasedDistribution.__init__` and the one-line docstrings on `LaserFrame._save` / `_save_dict` helpers."
  concise:
    score: 9
    weight: 1
    reason: "Validation logic is deduplicated into `_validation.py`; vectorized `distance()` and `competing_destinations()` eliminate prior O(N^2) Python loops; ruff + black linter config is present and enforced in CI; the only avoidable duplication is the near-identical `test_radiation_exclude_home`/`include_home` and `test_stouffer_*` test pairs that could be parameterized."
usability:
  simple:
    score: 8
    weight: 3
    reason: "Public surface clearly signposted in `__init__.py`, CLAUDE.md, and per-folder READMEs; major entry points have sensible defaults and validation raising `TypeError`/`ValueError`; quick-start one-liner workflows are shown in `docs/usage.rst`. Remaining issue: `stouffer` and `radiation` still require O(N) Python loops per source node, meaning users who try large inputs may encounter non-obvious slowness rather than a clean error."
  powerful:
    score: 8
    weight: 2
    reason: "All key parameters are exposed (k, a, b, c, delta, include_home, max_year, dtype, default); `LaserFrame` accepts arbitrary property names/dtypes, `PropertySet` supports multiple composition operators, distributions use `lru_cache` closures making them composable in Numba components. No documented subclassing hooks or plugin registry exist for custom migration models."
  performant:
    score: 8
    weight: 2
    reason: "Hot paths use Numba njit `parallel=True` (`_pyod`, `_pdod`, distribution kernels); `competing_destinations` and `distance` are now fully vectorized; a `pytest-benchmark` suite scaffolds regression coverage. `stouffer` and `radiation` still loop over source nodes in Python, and the new benchmark suite's baselines are flagged as 'starter placeholders' rather than CI-enforced gates."
  documented:
    score: 8
    weight: 2
    reason: "Multi-layered documentation: a root CLAUDE.md, per-folder READMEs, a fleshed-out `docs/usage.rst` quick-start, docstrings with runnable examples on all major public classes, and three Sphinx worked examples. Gaps: distribution docstrings show formulas but lack runnable `**Example**` blocks; `docs/migration.rst` has outstanding TODOs; no comprehensive parameter-choice tradeoff guide across the four migration models."
  accessible:
    score: 9
    weight: 1
    reason: "PyPI-published (15+ versions, `pip install laser-core`), MIT license, CHANGELOG.rst, CONTRIBUTING.rst, AUTHORS.rst, ReadTheDocs badge, GitHub issues link, and a repo-root CLAUDE.md for AI assistants. Falls one short of 10 because no MCP server or skills manifest is present and the maintainer contact (in `pyproject.toml`) is not surfaced in the README or CONTRIBUTING guide."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license confirmed at the repo root; no exposed secrets found via grep scan; all core dependencies (click, numpy, numba, matplotlib, pandas, h5py, geopandas, shapely) carry permissive licenses (BSD/MIT); CHANGELOG.rst, CONTRIBUTING.rst, and AUTHORS.rst are all present."
  reproducible:
    score: 10
    weight: 4
    reason: "`pyproject.toml` specifies lower-bounded version pins (>=) on all key dependencies; `uv.lock` provides full environment pinning; `random.py` exposes a seedable PRNG with deterministic behavior; semantic-versioning git tags exist through v1.0.2 and the package is confirmed published on PyPI (HTTP 200)."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.
- **Score delta vs. previous audit (84/100 → 86/100)**:
  - Improved: `clear` 8→9, `concise` 8→9, `simple` 7→8, `performant` 7→8, `reproducible` 9→10.
  - Held: `powerful` 8, `documented` 8, `accessible` 9, `compliant` 10.
  - Regressed: `correct` 9→8 — this re-review surfaced `assert` usages in `propertyset.py` and `utils.py` that the prior audit did not flag; addressing them in addition to the original `calc_capacity` fix would restore (and slightly exceed) the prior score.

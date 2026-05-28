# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 92/100
- **Status**: PASS
- **Date**: 2026-05-28
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 172s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 90/100 | 40% |
| Usability | 90/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **92/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 9/10 | 272 tests on 3-OS / 2-Python CI, bit-equivalence regression coverage on every vectorized migration kernel, `RE = 6371.0` now cited (Moritz 2000); residual gap: the `safety_multiplier = 1 + safety_factor * (sqrt(exp_mu_t) - 1)` heuristic in `utils.calc_capacity` has no derivation or citation. |
| clear | 9/10 | One-module-per-concern, `interrogate` at 97.9% (95% gate), Google-style docstrings everywhere; `propertyset.py` class docstring still uses RST `::` blocks rather than the project's Google/Markdown convention. |
| concise | 9/10 | No avoidable duplication; vectorized hot paths; `lru_cache` on distribution factories; structural repetition in `LaserFrame.describe()` is legitimate formatting variation. |
| simple | 9/10 | `dist.sample(dist.poisson, n=1000, lam=3.0)` and `build_network(model_fn, pops, distances, max_rowsum=...)` collapse common workflows into single calls; `_validation` errors are descriptive at every boundary; `docs/migration.rst` preamble still carries some informal TODO-style developer notes. |
| powerful | 9/10 | Numba-compatible composition (`mixture2` / `tick_modulated` / `node_modulated`), `**kwargs` passthrough in all migration models, `build_network` composes any conforming callable, worked `LaserFrame` / `PropertySet` / custom-migration extension examples in `docs/architecture.rst`. No concrete remaining gap. |
| performant | 9/10 | All five migration functions and `distance` fully vectorized; `numba.njit(parallel=True)` on sampling kernels; `pytest-benchmark` suite end-to-end runnable with a committed `.benchmarks/baseline.json`; CI `benchmarks.yml` runs the suite. Only remaining gap: workflow is still `continue-on-error: true`, so regressions are detected but not enforced. |
| documented | 9/10 | Per-folder READMEs at every major directory, runnable `docs/usage.rst` quick-start, `docs/migration.rst` Choosing-a-model + Performance, `docs/architecture.rst` Extension Points, `**Example**` blocks on every public distribution factory, `CLAUDE.md` + `.claude/skills/laser-core.md` for AI orientation. The informal TODO notes at the top of `docs/migration.rst` are the only content gap. |
| accessible | 9/10 | PyPI-published `laser-core` v1.0.2, MIT, `CHANGELOG.rst` / `CONTRIBUTING.rst` / `CODE_OF_CONDUCT.md` / `AUTHORS.rst` all present, README `Support and contact` section, `CLAUDE.md` + Claude Code skill manifest. No MCP server endpoint. |
| compliant | 10/10 | MIT license, no secrets, all runtime deps permissively licensed, all community files present. |
| reproducible | 10/10 | `>=` lower bounds on every runtime dep, `uv.lock` committed, deterministic seeding via `random.py` consumed throughout, semver tags through v1.0.2 on PyPI. |

`laser.core` jumps to **92/100** (from 89), with parallel +1 movement in Quality (90), Usability (90), and Safety holding perfect (100). The previous benchmark API bugs, missing READMEs, stale docstrings, and missing `CODE_OF_CONDUCT.md` are all now closed, and a baseline JSON has been committed under `.benchmarks/`. Every individual metric outside of Safety now sits at 9/10. The remaining gaps to a perfect 100 are small and concrete: one un-derived numerical heuristic in `calc_capacity`, the still-`continue-on-error` benchmark CI gate, a few `docs/migration.rst` preamble notes that read like dev TODOs, the lingering RST-style block in `PropertySet`'s class docstring, and the absence of a live MCP endpoint.

## Recommendations

1. **performant — Flip the benchmark CI workflow to enforcing** *(effort: quick; automated: yes)*
   The baseline at [.benchmarks/Linux-CPython-3.12-64bit/baseline.json](.benchmarks/) is committed and the workflow already runs the suite. Two edits in [.github/workflows/benchmarks.yml](.github/workflows/benchmarks.yml): remove `continue-on-error: true`, and replace the "advisory" run step with `pytest benchmarks/ --benchmark-only --benchmark-compare=.benchmarks/Linux-CPython-3.12-64bit/baseline.json --benchmark-compare-fail=min:25%`. This raises `performant` 9→10 outright. Suggested first threshold is `min:25%`; tighten to `min:15%` after observing a few PR runs to calibrate the noise floor.

2. **correct — Derive or cite the `safety_multiplier` formula in `calc_capacity`** *(effort: quick; automated: no — needs author intent)*
   In [src/laser/core/utils.py:70](src/laser/core/utils.py), `safety_multiplier = 1 + safety_factor * (sqrt(exp_mu_t) - 1)` lacks any derivation. Either add a comment explaining the rationale (it appears to interpolate between `1 + safety_factor * (sqrt(growth) - 1)` for stochastic variance amortization) or cite the reference it came from. Lifts `correct` 9→10.

3. **documented / simple — Clean up the informal TODO-style preamble in `docs/migration.rst`** *(effort: quick; automated: no)*
   The first ~15 lines of [docs/migration.rst](docs/migration.rst) contain notes that read like developer TODOs ("I'm aiming to do most of this with element-by-element numpy", "I have not tested nor written code to enforce conditions on the inputs", "maybe each function should start by adding epsilon to the diagonal"). These are interesting historical context but they now belong in commit messages or design docs, not the user-facing reference page. Move them to `notes.md` (or delete them entirely now that the work is done) and replace with a one-paragraph statement of what the module does. Lifts both `documented` and `simple` 9→10.

4. **clear — Convert `PropertySet`'s class docstring to Google/Markdown style** *(effort: quick; automated: yes)*
   In [src/laser/core/propertyset.py](src/laser/core/propertyset.py), the class docstring uses RST `::` directive blocks inconsistent with the project's Google/Markdown convention (which the rest of the codebase follows and which the docs/MkDocs renderer expects). Rewrite as a Google-style block with `Attributes:` and an `**Example**:` section. Lifts `clear` 9→10.

5. **accessible — Add a small MCP server alongside the existing Claude Code skill manifest** *(effort: medium; automated: no — design decision)*
   The skill manifest at [.claude/skills/laser-core.md](.claude/skills/laser-core.md) closed most of the AI-optimization gap. The Tier 1 anchor for `accessible` 10 specifically mentions both skills AND MCP. A small read-only MCP server (e.g., exposing the public API, the current `engineering_score.md`, and the migration model choice guide) would close the remaining point. Lifts `accessible` 9→10.

6. **powerful — Document the duck-typed extension protocol for migration models** *(effort: quick; automated: no)*
   `docs/architecture.rst` shows a custom migration model as a plain function with `(pops, distances, **kwargs) -> 2D array`. Tighten this into a one-paragraph "protocol" section explicitly listing what the input shapes/types are, what the diagonal/zero-handling convention is, and what `**kwargs` are conventionally accepted. This nudges `powerful` 9→10 by making the extension contract literal rather than implicit.

## Full Results

```yaml
project: /Users/christopherlorton/projects/laser-fresh/laser.core
tier: 1
overall_score: 92
failed: false
quality:
  correct:
    score: 9
    weight: 7
    reason: "272 tests covering all main workflows and edge cases with bit-equivalence pinning against loop references, CI/CD on 3 OS × 2 Python versions, migration models cite published references, and `RE = 6371.0` is now cited (Moritz 2000). The `safety_multiplier = 1 + safety_factor * (sqrt(exp_mu_t) - 1)` heuristic in `utils.calc_capacity` lacks a derivation or citation beyond the GBM comment that covers only `exp_mu_t`."
  clear:
    score: 9
    weight: 2
    reason: "Well-organized modular structure with `interrogate` gate at 95% (measured 97.9%), Google-style docstrings with Args/Returns/Raises/Example on all public APIs, and descriptive names throughout. `propertyset.py` class docstring uses RST `::` block formatting instead of the project's Markdown/Google convention."
  concise:
    score: 9
    weight: 1
    reason: "No avoidable duplication; hot paths use `@nb.njit(parallel=True)` and vectorized NumPy with `lru_cache` for kernel caching; ruff and black enforced in pre-commit and CI. The `describe()` method in `laserframe.py` repeats a structural pattern across scalars/vectors/others sections, but this is legitimate formatting variation."
usability:
  simple:
    score: 9
    weight: 3
    reason: "`dist.sample(dist.poisson, n=1000, lam=3.0)` and `build_network(model_fn, pops, distances, max_rowsum=...)` collapse common workflows into one-liners; `_validation` helpers produce descriptive `ValueError`/`TypeError` messages at every boundary. `docs/migration.rst` retains informal TODO-style developer notes in its preamble alongside the polished user-facing content."
  powerful:
    score: 9
    weight: 2
    reason: "Numba-compatible composition helpers (mixture2, tick_modulated, node_modulated) allow arbitrary distribution pipelines; migration models accept `**kwargs` and `build_network` composes any conforming model function with optional row normalization; `docs/architecture.rst` Extension Points section provides worked subclassing examples for `LaserFrame`, `PropertySet`, and custom migration models. The extension protocol is duck-typed rather than formally specified."
  performant:
    score: 9
    weight: 2
    reason: "All migration models are fully vectorized NumPy; distribution samplers use `@nb.njit(parallel=True)`. A `pytest-benchmark` suite with JIT warmup covers all hot paths, a committed `.benchmarks/Linux-CPython-3.12-64bit/baseline.json` enables regression detection, and CI runs the suite. The benchmarks CI workflow still retains `continue-on-error: true` so regressions are observed but not enforced."
  documented:
    score: 9
    weight: 2
    reason: "Per-folder READMEs at every major directory; `docs/usage.rst` quick-start, `docs/migration.rst` Choosing-a-model + Performance, `docs/architecture.rst` Extension Points; `interrogate` ≥95% gate with runnable `**Example**` blocks on major public functions. `CLAUDE.md` and `.claude/skills/laser-core.md` add a Tier-1 AI-orientation differentiator. The only documentation content that falls short of a comprehensive user guide is the informal TODO-style preamble at the top of `docs/migration.rst`."
  accessible:
    score: 9
    weight: 1
    reason: "Published on PyPI (`laser-core` v1.0.2, MIT); installation is `pip install laser-core` (one command). `LICENSE`, `CHANGELOG.rst`, `CONTRIBUTING.rst`, `CODE_OF_CONDUCT.md`, and `AUTHORS.rst` are all present, and `README.rst` provides maintainer emails and GitHub Issues for support. `CLAUDE.md` + `.claude/skills/laser-core.md` close most of the AI-optimization gap; no MCP server endpoint."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license confirmed at the repo root; no secrets, PII, or hardcoded credentials found anywhere in source; all runtime dependencies (click, numpy, numba, matplotlib, pandas, h5py, geopandas, shapely, uv) carry permissive BSD/MIT/Apache licenses."
  reproducible:
    score: 10
    weight: 4
    reason: "All runtime deps carry `>=` lower-bound pins in `pyproject.toml`, `uv.lock` provides a full lockfile, `laser.core.random.seed()` provides process-wide deterministic seeding for both NumPy and Numba contexts, semver git tags exist through v1.0.2, and the package is published on PyPI (HTTP 200)."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.
- **Score delta vs. previous audit (89 → 92, +3)**:
  - **Quality 90 (held)**: `correct` 9, `clear` 9, `concise` 9 — same scores but the gap items are different. The stale `distributions` API bug and the `LaserFrame` docstring inaccuracies that held `correct` and `clear` are now closed; new findings are the `calc_capacity` heuristic and the `PropertySet` docstring formatting.
  - **Usability 83 → 90 (+7)**: `simple` 8→9 (one-liner workflows landed), `performant` 7→9 (the broken benchmark suite + missing baseline are now fixed and committed), `documented` 9 held, `powerful` 9 held, `accessible` 9 held (with `CODE_OF_CONDUCT.md` now present).
  - **Safety 100 (held)**: no changes; both metrics already maxed.
- **Cheapest remaining lever to a perfect 100**: flipping the benchmark workflow from `continue-on-error: true` to enforcing (recommendation #1) — a literal 2-line edit now that the baseline is committed. Lifts `performant` 9→10 and projects **92 → ~93**. Then recommendations #2 / #3 / #4 / #6 each gain 0.5–1 more points, and a small MCP server (#5) closes the last gap.

# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 91/100
- **Status**: PASS
- **Date**: 2026-05-28
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 164s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 90/100 | 40% |
| Usability | 88/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **91/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 9/10 | 271 tests on 6-leg CI, bit-equivalence regression coverage on every vectorized migration kernel, all numeric constants cited (Moritz 2000 for `RE`, GBM derivation for `safety_multiplier`); the one deduction is that the benchmark `--benchmark-compare-fail` gate in CI is explicitly commented out, so perf regressions are observed but not enforced. |
| clear | 9/10 | Modular one-concern-per-file structure with Google-style docstrings on essentially all public APIs (interrogate 97.8%, 95% gate); minor: several `LaserFrame`/`PropertySet` methods use `Parameters:` instead of the project-mandated `Args:` header, and `PropertySet.to_dict` / `save` have one-liner docstrings without full structure. |
| concise | 9/10 | No avoidable duplication; vectorized hot paths with `@nb.njit(parallel=True)`; ruff+black via pre-commit. A small commented-out `multinomial` block in `distributions.py` is the only dead code. |
| simple | 9/10 | Three core classes + `build_network` + `distributions.sample` one-liners; thorough boundary validation. `LaserFrame.load_snapshot` capacity semantics (with `cbr`/`nt` together-or-not-at-all) is the only edge that could trip new users. |
| powerful | 9/10 | Migration-model protocol now formally documented; all four built-ins accept `**params`; `LaserFrame`/`PropertySet` subclassable with worked examples; new composition helpers `mixture2`/`tick_modulated`/`node_modulated`. Minor: no `mixture3+` for higher-order distribution mixtures and no whole-simulation-state snapshot helper. |
| performant | 8/10 | All five migration models + `distance` fully vectorized; Numba parallel kernels cover distributions and `KaplanMeierEstimator`; `pytest-benchmark` suite with a **committed baseline** under `.benchmarks/Linux-CPython-3.12-64bit/`. **Score regressed from 9 because** the benchmark CI workflow has the `--benchmark-compare-fail=min:25%` line **commented out**, so the gate is observation-only. `SortedQueue` benchmarks also only test at n=10k vs. the production scale of millions advertised in the class docstring. |
| documented | 9/10 | Per-folder READMEs at every level, runnable `**Example**` blocks on every public distribution and the three core classes, `docs/usage.rst` quick-start, `docs/migration.rst` Choosing-a-model + Performance characteristics, `docs/architecture.rst` Extension Points with formal migration-model protocol. Gaps: `docs/performance.rst` advises users to ask ChatGPT for component-tuning help and contains no concrete profiling data for `laser.core`'s own hot paths; HDF5 snapshot and `SortedQueue` workflows are not covered in depth. |
| accessible | 9/10 | PyPI v1.0.2, MIT, `LICENSE` / `CHANGELOG.rst` / `CONTRIBUTING.rst` / `CODE_OF_CONDUCT.md` / `AUTHORS.rst` all present, README `Support and contact` section, `CLAUDE.md` + `.claude/skills/laser-core.md` skill manifest. The only standard Tier-1 element missing is a live MCP server endpoint. |
| compliant | 10/10 | MIT license, no secrets, no PII in data CSVs, all runtime deps permissively licensed (BSD/MIT/Apache). |
| reproducible | 10/10 | `>=` lower bounds on every runtime dep, `uv.lock` committed, deterministic seeding via `random.py`, semver tags v0.5.1 → v1.0.2 on PyPI. |

**Net result:** Quality holds at 90/100, Safety holds at a perfect 100/100, and Usability is 88/100 (down 2 from the prior 90). The fix-pass that landed the `safety_multiplier` derivation, the `PropertySet` docstring rewrite, the `docs/migration.rst` cleanup, and the migration-model protocol formalization did everything it set out to do — those metrics still sit at 9/10 each. The single regression vs. the prior 92 audit is `performant` 9→8, surfaced by the auditor on closer inspection: the benchmark baseline IS committed, but the `--benchmark-compare-fail=min:25%` line in `.github/workflows/benchmarks.yml` is still commented out, so the gate observes but doesn't enforce. Flipping that one line (and removing `continue-on-error: true`) is the single largest remaining lever to a 95+/100 score.

## Recommendations

1. **performant 8→10 — Activate the benchmark CI gate** *(effort: quick; automated: yes)*
   In [.github/workflows/benchmarks.yml](.github/workflows/benchmarks.yml): (a) uncomment the `pytest benchmarks/ ... --benchmark-compare=.benchmarks/Linux-CPython-3.12-64bit/baseline.json --benchmark-compare-fail=min:25%` step, (b) delete the `continue-on-error: true` line, and (c) delete the now-obsolete TODO comment blocks. The baseline is already committed. This single change lifts `performant` 8→9 immediately; tightening to `min:15%` after a few PR runs to calibrate the noise floor projects 9→10.

2. **accessible 9→10 — Scaffold a small MCP server** *(effort: medium; automated: partial)*
   Add a small read-only MCP server under `mcp/laser_core_mcp/` exposing a few tools (`get_public_api`, `get_engineering_score`, `get_migration_choice_guide`). Register it via `.claude/mcp.json`. The static skill manifest at `.claude/skills/laser-core.md` already covers Claude Code; an MCP server primarily helps OTHER clients (Cursor, Cline, etc.). Marginal value is moderate; do it if you want broader AI-tool discoverability.

3. **clear 9→10 — Normalize remaining `Parameters:` to `Args:`** *(effort: quick; automated: yes)*
   Sweep `LaserFrame` (`sort`, `squash`, `add_*`) and `PropertySet` docstrings to use the project's `Args:` header consistently; expand the one-liner docstrings on `PropertySet.to_dict` and `PropertySet.save` to include `Args:` / `Returns:` blocks.

4. **correct 9→10 — Mirror the benchmark gate fix** *(effort: quick; automated: yes)*
   Same workflow edit as #1 directly closes the `correct` deduction the auditor flagged ("benchmark regression gate explicitly commented out"). One change, two metrics improved.

5. **performant 9→10 — Extend `SortedQueue` benchmarks to production scale** *(effort: quick; automated: yes)*
   The class docstring advertises support for "tens to hundreds of millions of agents" but [benchmarks/test_sortedqueue_benchmarks.py](benchmarks/test_sortedqueue_benchmarks.py) only tests at n=10k. Add a `test_sortedqueue_push_pop_1m` (or 100k as an interim) so the benchmark suite reflects the documented production scale.

6. **documented 9→10 — Replace `docs/performance.rst` ChatGPT pointer with concrete profiling data** *(effort: medium; automated: partial)*
   The performance guide currently refers users to ChatGPT for component-tuning help. Replace with concrete profiling traces / numbers (or a `py-spy`/`scalene` walkthrough) of `laser.core`'s own hot paths now that the benchmark suite is live. Also add a short HDF5 snapshot round-trip section and a `SortedQueue` push/peek/pop workflow section.

7. **powerful 9→10 — Add `mixtureN` and a full-simulation-state save/load** *(effort: medium; automated: yes for `mixtureN`)*
   Generalize `mixture2` to `mixture(*samplers, weights=...)` so multi-component mixtures don't require nested compositions. Optionally add a `save_simulation_state` / `load_simulation_state` pair that round-trips a `LaserFrame` + `PropertySet` + a list of associated arrays through one HDF5 file.

## Full Results

```yaml
project: /Users/christopherlorton/projects/laser-fresh/laser.core
tier: 1
overall_score: 91
failed: false
quality:
  correct:
    score: 9
    weight: 7
    reason: "271 tests pass across 11 files covering all modules, CI runs on 6 OS/Python combinations, benchmarks tracked in CI, all numeric constants are cited (Moritz 2000 for `RE`, GBM derivation for `safety_multiplier`, migration models cite Simini/Fotheringham/Stouffer), and the PRNG convention is enforced throughout. The benchmark regression gate (`--benchmark-compare-fail`) is explicitly commented out, so benchmark regressions do not yet fail CI."
  clear:
    score: 9
    weight: 2
    reason: "Modular one-concern-per-file structure with Google-style docstrings on essentially all public APIs (interrogate at 97.8%, 95% gate enforced in pre-commit). Minor inconsistency: several `LaserFrame` and `PropertySet` methods use `Parameters:` instead of the project-mandated `Args:` header, and short methods like `PropertySet.to_dict` / `save` have one-liner docstrings without the full Args/Returns structure."
  concise:
    score: 9
    weight: 1
    reason: "No avoidable duplication found; hot paths use Numba parallel kernels and vectorized NumPy; ruff + black enforced via pre-commit. The only minor issue is a commented-out `multinomial` block in `distributions.py` that is dead code but not pervasive."
usability:
  simple:
    score: 9
    weight: 3
    reason: "Three core classes (LaserFrame, PropertySet, SortedQueue) and all migration/distribution functions have clear, intuitive APIs with sensible defaults and thorough `ValueError`/`TypeError` validation; `build_network` and `distributions.sample` make common workflows one-liners. The one gap is that `LaserFrame.load_snapshot` requires `cbr` and `nt` together with somewhat complex capacity semantics that could trip up new users."
  powerful:
    score: 9
    weight: 2
    reason: "All model parameters are explicit kwargs, all four migration models follow an open `**params` protocol enabling custom models as first-class peers, `LaserFrame`/`PropertySet` are designed for clean subclassing with worked examples in `docs/architecture.rst`. Distributions lack a generalized `mixtureN` (only `mixture2` exists) and there is no built-in full-simulation-state save/load across frames."
  performant:
    score: 8
    weight: 2
    reason: "All five migration models + `distance` fully vectorized; Numba parallel kernels cover distributions and `KaplanMeierEstimator`; pytest-benchmark suite with a committed baseline at `.benchmarks/Linux-CPython-3.12-64bit/baseline.json` covers all hot paths. The benchmark CI workflow runs comparisons but the `--benchmark-compare-fail=min:25%` line is commented out (and `continue-on-error: true` is still in place), so regressions do not actually fail the build. `SortedQueue` push/pop benchmarks only test at n=10k rather than the production scale of millions advertised in the class docstring."
  documented:
    score: 9
    weight: 2
    reason: "Per-folder READMEs at every level, runnable `**Example**` blocks on every public distribution factory and the three core classes, `docs/usage.rst` runnable quick-start, `docs/migration.rst` model tradeoffs and performance characteristics, `docs/architecture.rst` formalizes the migration-model protocol and extension patterns, plus three end-to-end RST walkthroughs and two Jupyter notebooks. `docs/performance.rst` refers users to ChatGPT rather than providing profiling data for `laser.core`'s own hot paths, and HDF5 snapshot and `SortedQueue` workflows are not covered in depth."
  accessible:
    score: 9
    weight: 1
    reason: "Published on PyPI as `laser-core` v1.0.2 (one-command install), MIT license, all community files present (CHANGELOG.rst, CONTRIBUTING.rst, CODE_OF_CONDUCT.md, AUTHORS.rst), and a Support/contact section in `README.rst`. `CLAUDE.md` + `.claude/skills/laser-core.md` skill manifest provide AI-assistant orientation. The only missing element for a 10 is a formal MCP server endpoint."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license confirmed at the repo root; no secrets, PII, or restrictive-license dependencies found; all runtime dependencies (click, numpy, numba, matplotlib, pandas, h5py, geopandas, shapely, uv) carry BSD/MIT/Apache permissive licenses; CODE_OF_CONDUCT.md, CONTRIBUTING.rst, AUTHORS.rst, and CHANGELOG.rst are all present."
  reproducible:
    score: 10
    weight: 4
    reason: "All runtime dependencies carry `>=` lower bounds in `pyproject.toml`, a `uv.lock` lock file is present, the seedable PRNG in `random.py` controls process-wide reproducibility, semver git tags span v0.5.1 → v1.0.2, and the package is published on PyPI (HTTP 200)."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.
- **Score delta vs. previous audit (92 → 91, −1)**:
  - **Quality 90 (held)**: `correct` 9 / `clear` 9 / `concise` 9.
  - **Usability 90 → 88 (−2)**: `performant` 9→8 — the auditor on closer look saw that `.github/workflows/benchmarks.yml` keeps `--benchmark-compare-fail` commented out alongside `continue-on-error: true`, so the committed baseline is observed but not enforced. The previous 9 was generous; the current 8 is the honest reading. Everything else held.
  - **Safety 100 (held)**.
- **Top remaining lever**: recommendation #1 (activate the benchmark CI gate) is a literal 3-line edit and lifts `performant` 8→9 + closes the `correct` deduction simultaneously, projecting overall **91 → ~93**. Stacked with recommendations #3 (Parameters→Args sweep) and #5 (sortedqueue benchmark at production scale), the overall projects to **~94/100**. The remaining gap to a perfect 100 is the MCP server (#2), the `mixtureN` + snapshot helpers (#7), and the `docs/performance.rst` rewrite (#6) — all medium-effort, partially-automatable.

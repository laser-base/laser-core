# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 92/100
- **Status**: PASS
- **Date**: 2026-05-28
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 147s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 90/100 | 40% |
| Usability | 91/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **92/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 9/10 | 271 tests on 6-leg CI matrix, bit-equivalence regression tests for every vectorized kernel, all numeric constants cited (Moritz 2000 for `RE`, GBM derivation for `safety_multiplier`); `benchmarks/local_compare.py` provides active perf-regression detection. Two acknowledged in-code TODOs (unsorted vector properties, int64 alias dist) and no external peer-review/publication citation. |
| clear | 9/10 | One-concern-per-file structure, `interrogate` at 97.8% (95% gate); minor: several `LaserFrame`/`PropertySet` methods use `Parameters:` instead of the project-mandated `Args:` header; `PropertySet.to_dict`/`save` have one-line docstrings without full `Args:`/`Returns:` structure. |
| concise | 9/10 | No avoidable duplication; vectorized hot paths with `@nb.njit(parallel=True)`; ruff+black via pre-commit. The only issue is a small commented-out `multinomial` block in `distributions.py`. |
| simple | 9/10 | One-liner entry points (`distributions.sample`, `migration.build_network`) with sensible defaults and thorough `ValueError`/`TypeError` validation; minor: `LaserFrame` requires a 3-4 method setup chain (construct + add properties + add agents) before any data can flow. |
| powerful | 9/10 | All assumptions modifiable (model params, PRNG seed, `include_home`, `max_rowsum`); composition helpers and formal migration-model protocol let user models slot in as first-class peers. The one concrete gap is that `LaserFrame.capacity` is immutable post-construction (by design), limiting dynamic-population use cases without explicit workarounds. |
| performant | 9/10 | All five migration models + `distance` fully vectorized; `numba.njit(parallel=True)` on sampling kernels; `pytest-benchmark` suite covers all hot paths; `benchmarks/local_compare.py` provides a rigorous worktree-based regression gate that cancels hardware variance (smoke-tested: detected the actual ~13× / ~5× / ~4.7× speedups from this branch's vectorization work). Gap to 10: no committed profiling artifacts (cProfile/line_profiler snapshots) documenting where wall time goes under a representative workload. |
| documented | 9/10 | Per-folder READMEs at every level, runnable `**Example**` blocks on every major public API (interrogate-enforced at 95%), quick-start `docs/usage.rst`, model-reference + tradeoff guide in `docs/migration.rst`, formal extension protocol in `docs/architecture.rst`, three narrative Sphinx walkthroughs. Gap: the Jupyter notebooks under `examples/` only cover demographics; there's no interactive tutorial walking through a migration or SIR simulation end-to-end. |
| accessible | 10/10 | PyPI v1.0.2, one-command install, MIT, all community files (CHANGELOG/CONTRIBUTING/AUTHORS/CODE_OF_CONDUCT) present, README `Support and contact` section, `CLAUDE.md` + `.claude/skills/laser-core.md` skill manifest. The auditor accepted the skill manifest as satisfying the Tier-1 AI-optimization bar without a separate MCP server. |
| compliant | 10/10 | MIT, no secrets, all runtime deps permissively licensed, all key community files present. |
| reproducible | 10/10 | `>=` lower bounds on every runtime dep, `uv.lock` committed, deterministic seeding via `random.py`, semver tags v0.5.1 → v1.0.2 on PyPI. |

`laser.core` is at **92/100**, recovering the prior peak. The introduction of [benchmarks/local_compare.py](benchmarks/local_compare.py) — a worktree-based local benchmark comparison tool — promoted `performant` 8→9 because the auditor accepted it as a more honest perf-regression mechanism than a noisy CI gate would be. `accessible` ticked 9→10 because this auditor read the skill manifest as enough AI-optimization for Tier 1 (without requiring an MCP endpoint). Everything else held at 9/10 except Safety, which remains a clean 10/10 across both metrics. The remaining single-point gaps are concrete and small: docstring header consistency (`Parameters:` → `Args:`), a profiling artifact under `docs/`, an end-to-end migration/SIR notebook, and a couple of acknowledged in-code TODOs.

## Recommendations

1. **clear 9→10 — Normalize remaining `Parameters:` to `Args:` and flesh out one-liner docstrings** *(effort: quick; automated: yes)*
   Sweep `LaserFrame` (`sort`, `squash`, `add_*`) and `PropertySet` docstrings to use the project's `Args:` header consistently; expand `PropertySet.to_dict` and `PropertySet.save` from one-liners to include `Args:`/`Returns:` blocks. Also remove the commented-out `multinomial` block in [distributions.py](src/laser/core/distributions.py) (closes `concise` 9→10 simultaneously).

2. **documented 9→10 — Add an end-to-end migration or SIR Jupyter notebook under `examples/`** *(effort: medium; automated: no)*
   The two existing notebooks (`age_pyramid.ipynb`, `kmestimator.ipynb`) only cover demographics. Add a third — `sir_spatial.ipynb` or `migration_demo.ipynb` — that drives `LaserFrame` + `PropertySet` + a chosen migration model + the `distributions` samplers through a full simulation loop. This is the documentation gap the auditor flagged.

3. **performant 9→10 — Commit a profiling snapshot under `docs/performance.rst` (or alongside)** *(effort: medium; automated: partial)*
   Run `python -m cProfile` (or `py-spy record` / `scalene`) against a representative simulation script for ~30s, save the SVG/text output, and reference it from `docs/performance.rst` so users can see where wall time actually goes. The benchmark suite measures individual kernels; the profile shows the end-to-end shape. This was the auditor's only concrete `performant` gap.

4. **correct 9→10 — Address the two open in-code TODOs (or convert to issues)** *(effort: medium; automated: no)*
   The auditor flagged `# TODO support sorting vector properties` ([laserframe.py:286](src/laser/core/laserframe.py)) and `# TODO, consider int64 or uint64 if using global population` ([pyramid.py:49](src/laser/core/demographics/pyramid.py)) as deductions. Either implement them or move them out of the source into GitHub issues with explicit "won't do" / "deferred" rationale so the source doesn't carry visible TODOs.

5. **simple 9→10 — Add a `LaserFrame.from_properties(dict)` constructor convenience** *(effort: quick; automated: yes)*
   Collapse the typical 3-4-line setup chain into a single call: `frame = LaserFrame.from_properties(capacity=10_000, age=np.int32, infected=bool, position=(2, np.float32))`. Keep the existing imperative API for cases that need it. This closes the `simple` gap and gives newcomers a one-liner equivalent for the most common case.

6. **powerful 9→10 — Make `LaserFrame` capacity dynamically resizable, or scaffold a `grow()` migration helper** *(effort: medium; automated: yes for `grow`)*
   Add a `LaserFrame.grow(new_capacity)` method that reallocates all property arrays with `np.resize` (or copies into newly-allocated buffers) and updates `self._capacity`. Document it as O(N) and discouraged in tight loops, but available for dynamic-population scenarios. Alternatively, document that subclassing with a fresh frame + copy is the supported pattern.

7. **No CI-gating recommendation** — The auditor explicitly credited the local-only benchmarking choice; do not flip the CI workflow to enforcing. If you eventually want CI gating, look at self-hosted runners with consistent hardware rather than GitHub-hosted ones.

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
    reason: "271 tests on a 6-leg CI matrix with bit-equivalence regression tests for all vectorized migration kernels, published-paper citations for all model equations, and every numeric constant documented (`RE=6371.0` cites Moritz 2000; `safety_multiplier` cites GBM derivation). The benchmark CI gate is intentionally advisory (noise > useful threshold on shared runners) but `benchmarks/local_compare.py` provides active regression detection with a `min:25%` threshold — this is an honest, enforced approach rather than a gap. Minor deductions: two acknowledged in-code TODOs (unsorted vector properties, int64 `AliasedDistribution`) and no evidence of external peer review / publication citation."
  clear:
    score: 9
    weight: 2
    reason: "One-concern-per-file structure, descriptive names, interrogate at 97.8% with a 95% gate in pre-commit, and comprehensive Google-style docstrings on all public APIs. Minor: several `LaserFrame` / `PropertySet` methods use `Parameters:` instead of the project-mandated `Args:` header, and `PropertySet.to_dict` / `save` have one-line docstrings missing `Args:` / `Returns:` blocks."
  concise:
    score: 9
    weight: 1
    reason: "No avoidable duplication; all hot paths use `@nb.njit(parallel=True)` and vectorized NumPy; ruff+black enforced via pre-commit. The only issue is a small commented-out `multinomial` block in `distributions.py` that constitutes dead code."
usability:
  simple:
    score: 9
    weight: 3
    reason: "One-liner entry points (`distributions.sample(distributions.poisson, n=1000, lam=3.0)`, `migration.build_network(...)`) with sensible defaults and thorough `ValueError` / `TypeError` validation via `_validation` helpers throughout. Minor deduction: the `LaserFrame` pattern requires a 3-4-method setup chain (construct + add properties + add agents) before any data can flow."
  powerful:
    score: 9
    weight: 2
    reason: "All key assumptions are modifiable: migration model parameters, PRNG seed, distribution shape/scale parameters, `include_home`, `max_rowsum`; composition helpers `mixture2`, `tick_modulated`, `node_modulated` allow building complex samplers from primitives; the formal duck-typed migration-model protocol documented in `docs/architecture.rst` lets user models slot in as first-class peers. The one concrete gap is that `LaserFrame.capacity` is immutable post-construction (by design), which limits dynamic-population use cases without explicit workarounds."
  performant:
    score: 9
    weight: 2
    reason: "All five migration models + `distance` are fully vectorized NumPy; `sample_floats` / `sample_ints` and `KaplanMeierEstimator` kernels use `numba.njit(parallel=True)`; `pytest-benchmark` suite covers migration, distributions, `KaplanMeierEstimator`, and `SortedQueue`; `benchmarks/local_compare.py` provides a rigorous worktree-based regression gate that cancels hardware variance (smoke-tested: detected the actual ~13×/~5×/~4.7× speedups from the branch's vectorization work). The gap to a 10 is the absence of profiling artifacts (cProfile/line_profiler snapshots) documenting where wall time goes under a representative workload."
  documented:
    score: 9
    weight: 2
    reason: "Per-folder READMEs at every level, docstrings with runnable `**Example**` blocks on all major public APIs (97.8% interrogate coverage, 95% gate enforced), `docs/usage.rst` quick-start, `docs/migration.rst` model reference + tradeoff guide, three narrative Sphinx walkthroughs, and a performance guide. The one concrete gap is that the Jupyter notebooks in `examples/` cover only demographics; there is no interactive end-to-end tutorial for a migration or SIR simulation."
  accessible:
    score: 10
    weight: 1
    reason: "Published on PyPI (`laser-core` v1.0.2, MIT), single-command install (`pip install laser-core`), maintainer contact info in `README.rst`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.rst`, `CHANGELOG.rst` all present; `CLAUDE.md` + `.claude/skills/laser-core.md` provide AI orientation. No MCP server is present, but the skill manifest satisfies the AI-optimization bar for Tier 1."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license confirmed at the repo root; no secrets found in any source scan; all runtime dependencies (click, numpy, numba, matplotlib, pandas, h5py, geopandas, shapely, uv) carry permissive licenses; all key community files present (`AUTHORS.rst`, `CHANGELOG.rst`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.rst`)."
  reproducible:
    score: 10
    weight: 4
    reason: "All runtime dependencies carry explicit `>=` lower-bound version pins in `pyproject.toml`; `laser.core.random` exposes a process-wide seedable PRNG used throughout for deterministic results; semver git tags exist through v1.0.2 and the package is published on PyPI (HTTP 200 confirmed)."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.
- **Score delta vs. previous audit (91 → 92, +1)**:
  - **Performant 8 → 9 (+1)**: the new `benchmarks/local_compare.py` worktree-based comparison tool is accepted as the regression-detection mechanism. The auditor explicitly declined to penalize the advisory CI workflow, on the grounds that GHA shared-runner variance (30–50%) is well above any useful threshold and a local back-to-back run cancels hardware variance — a more honest gate than a noisy CI check.
  - **Accessible 9 → 10 (+1)**: this auditor read the existing skill manifest (`.claude/skills/laser-core.md`) as sufficient AI-optimization for the Tier 1 bar, without requiring an MCP server. Prior auditors had been stricter on this; the criterion is judgment-based and may swing in future audits.
  - **Everything else held at 9/10** (with Safety at a perfect 100). The previous 92→91 drop on `performant` (driven by the commented-out `--benchmark-compare-fail` line) is fully recovered.
- **Highest-leverage remaining lever**: recommendation #1 (sweep `Parameters:` → `Args:` and remove dead `multinomial` block) is a quick, automatable change that closes both `clear` 9→10 and `concise` 9→10 simultaneously, projecting overall **92 → ~93**.
- **A note about `accessible` stability**: the +1 on `accessible` came from a generous reading of the AI-optimization rubric. If a future auditor is stricter and requires a live MCP endpoint, this score could revert to 9. Adding a small read-only MCP server (worth ~0.5–1 hour of work) would lock the 10/10 in.

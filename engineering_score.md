# Project Engineering Score

- **Project**: `/Users/christopherlorton/projects/laser-fresh/laser.core`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Overall Score**: 86/100
- **Status**: PASS
- **Date**: 2026-05-22
- **Version**: idm-eng-plugin:eng-quality-checker v1.3_2026.04.13
- **Time spent**: 147s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 83/100 | 40% |
| Usability | 81/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **86/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 8/10 | 11-file test suite (~4,000 lines) on 3-OS × 2-Python CI, bit-equivalence regression tests guard the vectorized migration kernels, all API-boundary `assert`s have been replaced with `TypeError`/`ValueError`; the `100 / tot_pop` magic number in `spatialpops.py` is still unexplained. |
| clear | 9/10 | All public APIs have Google-style docstrings with Args/Returns/Raises/Example sections, the new `_cumulative_at_or_closer_2d` helper has algorithmic documentation, and ruff+black are enforced; one residual issue is that the `stouffer` docstring shows the radiation formula in its element-by-element block. |
| concise | 9/10 | Migration models are fully vectorized; shared validation lives in `_validation.py`; `_cumulative_at_or_closer_2d` is cleanly factored out and reused by both `stouffer` and `radiation`; no significant copy-paste detected. |
| simple | 8/10 | Public surface clearly enumerated in `__init__.py` and `docs/usage.rst`, all boundaries validated with informative errors; common multi-step spatial workflows (build network → row-normalize → apply) still require 3–4 line chains rather than a single convenience call. |
| powerful | 8/10 | All model parameters exposed; `PropertySet` composes with `+=`/`<<=`/`|=`; the new `docs/architecture.rst` Extension Points section shows subclassing and custom-model patterns; no registry/plugin pattern for migration models and no built-in sampler composition (mixture/time-varying) without raw Numba. |
| performant | 8/10 | `gravity`, `competing_destinations`, `stouffer`, `radiation`, and `distance` are fully vectorized; Numba `parallel=True` on hot kernels; `pytest-benchmark` suite ships with a CI workflow. The CI workflow is intentionally `continue-on-error: true` until baselines are committed, so regressions aren't actually gated yet. |
| documented | 8/10 | Per-folder READMEs, fleshed-out `docs/usage.rst` quick-start, `docs/architecture.rst` Extension Points, `docs/migration.rst` Performance section, runnable `**Example**` blocks on all distribution factories, and three Sphinx examples (SIR, vital-dynamics, spatial); the architecture glossary still has a stub `**Patch** Something...` entry and there is no model-choice tradeoff guide. |
| accessible | 9/10 | PyPI-published (15+ versions, `pip install laser-core`), MIT license, `CHANGELOG.rst`/`CONTRIBUTING.rst`/`AUTHORS.rst`, ReadTheDocs, and a repo-root `CLAUDE.md` for AI-assistant orientation; no MCP server or skills manifest, and maintainer contact is not surfaced in the README. |
| compliant | 10/10 | MIT license, no secrets, all runtime deps (click, numpy, numba, matplotlib, pandas, h5py, geopandas, shapely, uv) carry permissive licenses, and all key community files are present. |
| reproducible | 10/10 | `>=` lower bounds on every key runtime dep in `pyproject.toml`, `uv.lock` committed, fully documented seedable PRNG in `random.py`, semver tags through v1.0.2, package on PyPI (HTTP 200). |

`laser.core` holds at 86/100 after this audit pass: Safety is now perfect (100/100), Quality is 83/100 with the assert-sweep complete and all docstring gaps closed, and Usability sits at 81/100 with all four migration models fully vectorized and a benchmark CI workflow scaffolded. The four remaining single-point gaps are: the `100/tot_pop` magic number in `spatialpops.py`, the misstated element-by-element formula in the `stouffer` docstring, the placeholder benchmark CI baseline (`continue-on-error: true`), and small documentation polish items (stub glossary entry, missing model-choice tradeoff guide, missing MCP/skills manifest).

## Recommendations

1. **clear — Fix the `stouffer` docstring formula** *(effort: quick; automated: no)*
   In [src/laser/core/migration.py](src/laser/core/migration.py), the element-by-element formula in the `stouffer` docstring currently displays `network[i,j] = k * p_i * p_j / ((p_i + sum_k p_k)(p_i + p_j + sum_k p_k))` (the radiation model). Replace with Stouffer's actual formula: `network[i,j] = k * p_i^a * (p_j / sum_{k in Omega(i,j)} p_k)^b`. Also remove the duplicate `AssertionError` mentions in the dunder docstrings if any remain (rg confirms they were updated, but worth re-grepping).

2. **correct — Explain or replace the `100 / tot_pop` magic number** *(effort: quick; automated: no — needs domain knowledge)*
   In [src/laser/core/demographics/spatialpops.py](src/laser/core/demographics/spatialpops.py) (around line 60), either add a comment citing the source of `100` (what does it represent demographically — a per-mille rate cap? a max-share fraction?) or lift it to a named module-level constant / function parameter with a default.

3. **performant — Commit a benchmark baseline and flip the CI workflow to enforcing** *(effort: medium; automated: partial)*
   Run `pytest benchmarks/ --benchmark-only --benchmark-save=main` on a clean main commit on the canonical Linux 3.12 runner, commit the resulting JSON under `.benchmarks/Linux-CPython-3.12-64bit/main.json`, then in [.github/workflows/benchmarks.yml](.github/workflows/benchmarks.yml) remove `continue-on-error: true` and switch the run command to `--benchmark-compare=...benchmarks/.../main --benchmark-compare-fail=min:25%`. Update [benchmarks/README.md](benchmarks/README.md) to remove the "starter placeholders" caveat.

4. **documented — Fill in the architecture glossary stub and write a model-choice tradeoff guide** *(effort: medium; automated: no)*
   Replace the `**Patch** Something...` placeholder in [docs/architecture.rst](docs/architecture.rst). Add a short new doc (or section under `docs/migration.rst`) comparing the four migration models — when to prefer each, parameter-range intuitions, and numerical edge cases to watch for. This is the single highest-impact remaining item for `documented`.

5. **simple — Add a one-liner convenience for the common spatial-network workflow** *(effort: medium; automated: no)*
   Wrap the typical "build network → row-normalize → apply" sequence into a single helper such as `laser.core.migration.build_network(model, pops, distances, *, max_rowsum=None, **model_kwargs) -> np.ndarray`. Document it in [docs/usage.rst](docs/usage.rst). Keep the underlying low-level functions in place for advanced users.

6. **powerful — Add a sampler-composition helper to `distributions`** *(effort: medium; automated: no)*
   Expose a `distributions.mixture(p, samplers)` or `distributions.time_varying(fn_of_tick)` factory so users can build composite distributions without writing raw `@nb.njit` closures. Examples already exist as comments in [src/laser/core/distributions.py](src/laser/core/distributions.py); productize the pattern.

7. **accessible — Add an MCP server or skills manifest, and surface maintainer contact in the README** *(effort: medium; automated: no)*
   The repo-root [CLAUDE.md](CLAUDE.md) closed most of this gap; add a small `.claude/skills/` bundle or MCP server description so AI assistants can be steered toward canonical idioms. Also surface the maintainer email (currently in `pyproject.toml`) and a "Getting help" link in the README.

8. **clear / documented — Add `interrogate` to pre-commit and a docstring-coverage gate** *(effort: quick; automated: yes)*
   Configure `interrogate --fail-under=95` (or similar) in [.pre-commit-config.yaml](.pre-commit-config.yaml) to prevent docstring regressions and surface remaining gaps quickly. This is a small force-multiplier rather than a direct score lever, but it locks in the recent docstring-quality gains.

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
    reason: "Comprehensive test suite (11 files, ~4,000 lines) with CI/CD on 3 OS × 2 Python versions plus advisory benchmarks, vectorization regression tests with bit-equivalence checks, and proper `TypeError`/`ValueError` at all API boundaries; the remaining gap is the unexplained `100 / tot_pop` magic number in spatialpops.py line 60 (no comment or citation)."
  clear:
    score: 9
    weight: 2
    reason: "All public APIs have Google-style docstrings with Parameters/Returns/Raises/Examples sections, descriptive module-level docstrings, and a ruff+black style config in pyproject.toml; the minor gap is that the `stouffer` docstring shows the radiation formula in its element-by-element block, creating slight ambiguity about the model equation."
  concise:
    score: 9
    weight: 1
    reason: "Production code is fully vectorized (no Python loops in migration models), shared validation helpers in `_validation.py` eliminate boilerplate, and `_cumulative_at_or_closer_2d` is cleanly factored out to serve both stouffer and radiation; no significant copy-paste or dead code detected."
usability:
  simple:
    score: 8
    weight: 3
    reason: "Public API is clearly enumerated in `__init__.py` and `docs/usage.rst` with a runnable quick-start; all boundaries validate via `_validation.py` raising `TypeError`/`ValueError` with informative messages. Multi-step spatial workflows (network build → normalize → apply) still require 3–4 line chains rather than a single convenience call."
  powerful:
    score: 8
    weight: 2
    reason: "All migration model parameters exposed; `PropertySet` supports `+=`/`<<=`/`|=`; `LaserFrame` subclassing and custom-migration patterns are documented in `docs/architecture.rst`. Gaps: no sampler composition helper for `distributions` and no plugin/registry pattern for migration models — extension is by convention only."
  performant:
    score: 8
    weight: 2
    reason: "All four migration models and `distance` are fully vectorized NumPy with no Python loops over nodes; `SortedQueue` and distributions use `numba.njit`; a `pytest-benchmark` suite plus CI workflow exists. The workflow is intentionally `continue-on-error: true` until baselines are committed, so regressions aren't actually gated yet."
  documented:
    score: 8
    weight: 2
    reason: "Per-folder READMEs, `docs/usage.rst` quick-start, `docs/architecture.rst` Extension Points with subclassing examples, `docs/migration.rst` Performance section, runnable `**Example**` blocks on all distributions, and three Sphinx walkthroughs (SIR, vital-dynamics, spatial). Gaps: the architecture glossary has a `**Patch** Something...` stub and there is no model-choice tradeoff guide."
  accessible:
    score: 9
    weight: 1
    reason: "PyPI-published (15+ versions, `pip install laser-core`), MIT license, `CHANGELOG.rst`/`CONTRIBUTING.rst`/`AUTHORS.rst`, ReadTheDocs, and a repo-root `CLAUDE.md` for AI-assistant orientation. A score of 10 is held back by the absence of an MCP server or skills manifest and the maintainer contact not being surfaced in the README."
safety:
  compliant:
    score: 10
    weight: 6
    reason: "MIT license present at the repo root; no hardcoded secrets detected; all runtime dependencies (click, numpy, numba, matplotlib, pandas, h5py, geopandas, shapely, uv) carry permissive licenses; CHANGELOG.rst, CONTRIBUTING.rst, and AUTHORS.rst are all present."
  reproducible:
    score: 10
    weight: 4
    reason: "All key runtime dependencies have `>=` lower-bound pins in `pyproject.toml`; `uv.lock` committed; `random.py` exposes a documented seedable PRNG; semver git tags through v1.0.2; package published on PyPI (HTTP 200 confirmed)."
```

## Notes

- **General scoring principle**: If no specific improvements can be identified for a metric, score 10/10. If scoring below 10, always list the specific improvements that would raise the score. Don't dock points for theoretical issues — only for concrete, observable problems.
- **Score delta vs. previous audit (86 → 86)**: Net-flat. The recent improvement pass closed all of the previously-flagged gaps in `clear`, `correct` (asserts), `documented` (distribution examples + architecture/migration sections), `performant` (stouffer/radiation vectorized + benchmark scaffold), and `simple` (quick-start) — but auditors now surface a few small additional issues that hold each metric at its prior score:
  - The `stouffer` docstring formula bug was flagged (and is a quick fix).
  - The benchmark CI workflow is `continue-on-error: true` — by design, until a baseline is committed.
  - The architecture glossary has a stub `**Patch** Something...` entry.
  - No model-choice tradeoff guide.
  - No sampler-composition helper.
  - No MCP/skills manifest.
  - The `100 / tot_pop` magic number is still unexplained.
  - These are all addressable in roughly 1–4 hours of focused work; each item closes one of the open single-point gaps.

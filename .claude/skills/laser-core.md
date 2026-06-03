---
name: laser-core
description: |
  Orientation for AI coding assistants working in the `laser-core` repository.
  Loads the canonical public API surface, the dev workflow (tests, docs, benchmarks),
  the validation pattern, and the numerical-correctness invariants that distinguish
  laser-core from a typical Python library. Use when answering questions about the
  laser-core codebase or editing files under it.
---

# laser-core skill

This skill is a compact, AI-readable companion to [`CLAUDE.md`](../../CLAUDE.md).
It exists so Claude Code (and any other tool that supports skills/MCP) can quickly
load the conventions that govern this repository before making changes.

The authoritative source remains `CLAUDE.md`. If anything in this file disagrees
with `CLAUDE.md`, follow `CLAUDE.md`.

## What this package is

`laser-core` is the foundation library of the **LASER** (Light Agent Spatial
modeling for ERadication) toolkit. It provides a Numba-accelerated agent data
structure (`LaserFrame`), a parameter set (`PropertySet`), a priority queue
(`SortedQueue`), and a small set of demography, migration, distribution, and
PRNG utilities used by all other LASER packages.

## Canonical public surface

All of the following are re-exported from `laser.core.__init__`:

- **Classes**: `LaserFrame`, `PropertySet`, `SortedQueue`
- **Submodules**: `distributions`, `migration`, `random`, `demographics`, `utils`
- **Sub-API**: `demographics.AliasedDistribution`, `demographics.KaplanMeierEstimator`,
  `demographics.load_pyramid_csv`

Anything starting with `_` (e.g. `_validation`) is package-private — do not
import from it outside the package.

## Dev workflow

The project's virtualenv lives in `.venv`. Use `python3` (never `python`).

```bash
uv pip install -e ".[dev,test,docs]"     # install in editable mode
.venv/bin/python3 -m pytest tests/        # run the unit tests
pytest benchmarks/ --benchmark-only       # run performance regression cases
cd docs && make html                      # build the Sphinx docs
pre-commit run --all-files                # run ruff + black + interrogate
```

## Required conventions

Apply these unless the user explicitly asks otherwise.

- **Validation at API boundaries**: raise `ValueError`/`TypeError`. Never bare
  `assert` (it is suppressed under `python -O`). Use the shared helpers in
  `laser.core._validation` (`_is_instance`, `_is_dtype`, `_has_shape`,
  `_has_dimensions`, `_has_values`).
- **Random number generation**: acquire the PRNG via
  `laser.core.random.prng()`. Never call `np.random.*` directly. Reproducibility
  hinges on `laser.core.random.seed()` controlling the whole process.
- **Numerical correctness**: migration models in `src/laser/core/migration.py`
  cite published references in their docstrings. Do not change their numerical
  behavior without a regression test demonstrating bit-equivalence (within float
  tolerance) to the prior implementation. The
  `TestMigrationVectorizationRegression` class in `tests/test_migration.py` is
  the canonical pattern.
- **Performance**: hot paths use `numba.njit(parallel=True)` and vectorized
  NumPy. Avoid Python-level loops over agents or destinations. When you touch a
  hot path, add a `pytest-benchmark` case under `benchmarks/`.
- **Docstrings**: Google style; format for MkDocs/Markdown (not RST). Include
  `Args`, `Returns`, `Raises`, and a runnable `**Example**` block where it adds
  clarity. Cross-reference symbols with `[name][module.path.name]`.
- **Style**: `black` + `ruff` with `line-length = 140`. Prefer double quotes.
  Prefer `pathlib.Path` over `os.path`. Enforced via pre-commit.
- **Python versions**: support `>=3.9,<3.15`; CI tests against 3.10 and 3.14.

## Useful entry-point patterns

When the user asks for an example of any of these, use this canonical idiom:

- **Build and apply a migration network**: see
  `laser.core.migration.build_network(model_fn, pops, distances, *, max_rowsum=None, **kwargs)`.
- **Sample from a distribution into a preallocated buffer**: build a sampler
  via the relevant factory (e.g. `distributions.poisson(lam=3.0)`), allocate
  `out = np.empty(N, dtype=np.int32)`, then `distributions.sample_ints(sampler, out)`.
- **Compose distributions**: `distributions.mixture2`, `tick_modulated`,
  `node_modulated` — all return Numba-compatible samplers usable inside
  `sample_floats`/`sample_ints`.
- **Subclass `LaserFrame` or write a custom migration model**: see the
  "Extension Points" section in `docs/architecture.rst`.

## What NOT to change without explicit user permission

- The numerical kernels in `src/laser/core/migration.py`,
  `src/laser/core/demographics/kmestimator.py`, and
  `src/laser/core/distributions.py` are validated against published references
  or scipy. Behavior changes need a regression test.
- The C extension under `src/laser/core/extension.py` /
  `src/laser/core/_extension.c`.
- The CI matrix in `.github/workflows/github-actions.yml`.

## Where to look for more

- `CLAUDE.md` — the full project orientation (this skill is a subset).
- `docs/architecture.rst` — design principles, layout, extension points.
- `docs/migration.rst` — model-choice tradeoffs and performance characteristics.
- `docs/usage.rst` — runnable quick-start tour of the public API.
- `engineering_score.md` — the most recent automated quality audit.
- `CHANGELOG.rst` — the "Unreleased" section is the running log of recent edits.

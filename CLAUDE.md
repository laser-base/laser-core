# CLAUDE.md — `laser-core` project conventions

This file orients AI assistants (Claude Code, Cursor, etc.) and human contributors
to how `laser-core` is organized, how to run its tests and docs, and which APIs
are considered the public surface.

## What this package is

`laser-core` is the foundation library of the **LASER** (Light Agent Spatial
modeling for ERadication) toolkit: a high-performance, Numba-accelerated agent
data structure and a small set of demography, migration, and distribution
utilities used by all other LASER packages. See `README.rst` and `docs/` for
a longer description.

## Public surface

The top-level package re-exports the canonical entry points:

- Classes — `LaserFrame`, `PropertySet`, `SortedQueue`
- Submodules — `distributions`, `migration`, `random`, `demographics`, `utils`
- Sub-API — `demographics.AliasedDistribution`, `demographics.KaplanMeierEstimator`,
  `demographics.load_pyramid_csv`

Anything starting with `_` (e.g. `_validation`) is package-private; do not
import from it outside the package.

## Repository layout

- `src/laser/core/` — library code (one module per concern)
  - `laserframe.py`, `propertyset.py`, `sortedqueue.py` — core data structures
  - `migration.py` — gravity, radiation, Stouffer, competing-destinations, distance
  - `distributions.py` — Numba-accelerated samplers
  - `random.py` — process-wide seedable PRNG
  - `demographics/` — population pyramids, Kaplan-Meier estimator, spatial populations
  - `utils.py` — capacity estimation and spatial-grid helpers
  - `_validation.py` — package-private input-validation helpers
- `tests/` — pytest-based test suite (one `test_<module>.py` per module)
- `benchmarks/` — `pytest-benchmark` cases for hot paths (kept distinct from `tests/`)
- `docs/` — Sphinx documentation, hosted on ReadTheDocs
- `examples/` — Jupyter notebooks and CSV inputs for tutorials
- `.github/workflows/` — CI matrix on Linux / macOS / Windows × Python 3.10 and 3.14

## How to develop

Working in the project's virtualenv (.venv):

- **Install in editable mode**: `uv pip install -e ".[dev,test,docs]"`
- **Run tests**: `pytest` (or `tox -e py310` / `tox -e py314` for the CI matrix)
- **Run benchmarks**: `pytest benchmarks/ --benchmark-only`
- **Build docs**: `cd docs && make html`
- **Lint**: `pre-commit run --all-files` (runs `ruff` and `black`)
- **Release**: `bumpversion patch|minor|major`, then push the resulting tag

## Conventions

- **Python version**: support `>=3.9,<3.15`; CI tests against 3.10 and 3.14.
- **Style**: `black` + `ruff` with `line-length = 140`. Enforced via pre-commit.
- **Docstrings**: Google style; format for MkDocs / Markdown rendering (not RST).
  Include `Args`, `Returns`, `Raises`, and a runnable `**Example**` block where
  it adds clarity. Reference other symbols with `[name][module.path.name]`.
- **Quoting**: prefer double quotes unless the string already contains double quotes.
- **Paths**: prefer `pathlib.Path` over `os.path` whenever feasible.
- **Validation at API boundaries**: raise `ValueError` / `TypeError` (do not use
  bare `assert`, which is suppressed under `python -O`). `assert` is reserved
  for internal consistency checks. The shared helpers in
  `laser.core._validation` (`_is_instance`, `_is_dtype`, `_has_shape`,
  `_has_dimensions`, `_has_values`) are the preferred mechanism.
- **Numerical correctness**: migration models cite published references in their
  module docstrings. Do not change the numerical behavior of these kernels
  without a test demonstrating bit-equivalence (within float tolerance) to the
  prior implementation.
- **Random number generation**: never call `np.random.*` directly. Acquire the
  PRNG via `laser.core.random.prng()` so that `laser.core.random.seed()`
  controls reproducibility for the whole process.
- **Performance**: hot paths use `numba.njit(parallel=True)` and vectorized NumPy.
  Avoid Python-level loops over agents or destinations. The benchmark suite
  guards against regressions; add a new case when you touch a hot path.

## Tier-1 expectations

This project is held to the IDM Tier 1 engineering quality bar (library / DPG).
See `engineering_score.md` for the most recent automated audit and the
prioritized recommendation list.

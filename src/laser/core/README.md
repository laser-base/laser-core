# `laser.core` — package source

This directory contains the implementation of the `laser-core` library. New
modules should follow the conventions listed in the top-level `CLAUDE.md`.

## Public modules

| Module | What it provides |
| --- | --- |
| `laserframe.py` | `LaserFrame` — dynamically allocated agent / patch storage with scalar, vector, and array properties. |
| `propertyset.py` | `PropertySet` — composable parameter bag (`+=`, `<<=`, `|=`). |
| `sortedqueue.py` | `SortedQueue` — Numba-backed fixed-capacity priority queue. |
| `migration.py` | `gravity`, `radiation`, `stouffer`, `competing_destinations`, `distance`, `row_normalizer`. |
| `distributions.py` | Numba-accelerated samplers (constant, uniform, poisson, ...). |
| `random.py` | Process-wide PRNG with `seed()` / `prng()`; acquire all randomness through this module. |
| `utils.py` | `calc_capacity`, `grid`, `initialize_population` helpers. |
| `extension.py` | C extension loader (`compiled`). |
| `demographics/` | Population pyramids and Kaplan-Meier age-at-death utilities. |

## Package-private modules

| Module | What it provides |
| --- | --- |
| `_validation.py` | Shared input-validation helpers (`_is_instance`, `_is_dtype`, `_has_shape`, `_has_dimensions`, `_has_values`). Internal — do not import from outside the package. |

## Reference documentation

Long-form references live in `docs/`:

- `docs/architecture.rst` — overall design and data flow.
- `docs/migration.rst` — mathematical formulation of each migration model.
- `docs/kmestimator.rst` — Kaplan-Meier estimator reference.
- `docs/performance.rst` — performance and parallelization guidance.

See `docs/usage.rst` for a runnable quick-start tour of the public API.

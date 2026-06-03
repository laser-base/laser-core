# `laser.core` — package source

This directory contains the implementation of the `laser-core` library. New
modules should follow the conventions listed in the top-level `CLAUDE.md`.

## Public modules

| Module | What it provides |
| --- | --- |
| `laserframe.py` | `LaserFrame` — dynamically allocated agent / patch storage with scalar, vector, and array properties. |
| `propertyset.py` | `PropertySet` — composable parameter bag (`+=`, `<<=`, `|=`). |
| `sortedqueue.py` | `SortedQueue` — Numba-backed fixed-capacity priority queue. |
| `migration.py` | `gravity`, `radiation`, `stouffer`, `competing_destinations`, plus `distance`, `row_normalizer`, and the `build_network(model_fn, pops, distances, *, max_rowsum=None, **kwargs)` convenience composer. |
| `distributions.py` | Numba-accelerated distribution factories (`beta`, `binomial`, `constant_float`, `constant_int`, `exponential`, `gamma`, `logistic`, `lognormal`, `negative_binomial`, `normal`, `poisson`, `uniform`, `weibull`), composition helpers (`mixture2`, `tick_modulated`, `node_modulated`), and the `sample(fn_or_factory, n, *, dtype=None, tick=0, node=0, out=None, **factory_kwargs)` one-liner that wraps allocation + dispatch to `sample_floats` / `sample_ints`. |
| `random.py` | Process-wide PRNG with `seed()` / `prng()`; acquire all randomness through this module. |
| `utils.py` | `calc_capacity`, `grid` (uses the LASER-wide PRNG by default), `initialize_population`. |
| `extension.py` | C extension loader (`compiled`). |
| `demographics/` | Population pyramids (`AliasedDistribution`, `load_pyramid_csv`), Kaplan-Meier age-at-death utilities (`KaplanMeierEstimator`), and spatial-population distributions (`distribute_population_skewed`, `distribute_population_tapered`). See [`demographics/README.md`](demographics/README.md). |

## Package-private modules

| Module | What it provides |
| --- | --- |
| `_validation.py` | Shared input-validation helpers (`_is_instance`, `_is_dtype`, `_has_shape`, `_has_dimensions`, `_has_values`). Internal — do not import from outside the package. |

## Reference documentation

Long-form references live under [`docs/`](../../../docs/):

- `docs/architecture.rst` — overall design, layout, and the "Extension Points" patterns for subclassing `LaserFrame`, extending `PropertySet`, and writing custom migration models.
- `docs/migration.rst` — mathematical formulation of each migration model, the "Choosing a model" tradeoff guide, and performance characteristics.
- `docs/kmestimator.rst` — Kaplan-Meier estimator reference.
- `docs/performance.rst` — performance and parallelization guidance.
- `docs/usage.rst` — runnable quick-start tour of the public API.

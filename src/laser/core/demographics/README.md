# `laser.core.demographics`

Population-pyramid, survival-curve, and spatial-population utilities used to
initialize, age, and attrit agent populations in LASER simulations. All
randomness goes through the LASER-wide PRNG (see `laser.core.random.seed` /
`laser.core.random.prng`), so seeding controls reproducibility end-to-end.

## Modules

| Module | What it provides |
| --- | --- |
| `pyramid.py` | `AliasedDistribution` (Vose alias sampler — `O(1)` per draw) and `load_pyramid_csv` (loader for UN-format age pyramids). |
| `kmestimator.py` | `KaplanMeierEstimator` — predict the year-of-death (`predict_year_of_death`) and day-of-death (`predict_age_at_death`) from a cumulative-deaths source; underlying kernels use `numba.njit(parallel=True)`. |
| `spatialpops.py` | `distribute_population_skewed(tot_pop, num_nodes, frac_rural=0.3, min_pop=1000)` — one urban node plus an exponentially-tapered rural distribution with a guaranteed per-node floor. `distribute_population_tapered` — smooth power-law taper. |

## Re-exports from `laser.core.demographics`

`AliasedDistribution`, `KaplanMeierEstimator`, `load_pyramid_csv` are re-exported at the package level.

## Reference documentation

- `docs/pyramids.rst` — using population pyramids for initialization.
- `docs/kmestimator.rst` — Kaplan-Meier estimator reference and example.
- `docs/vdexample.rst` — vital-dynamics worked example.

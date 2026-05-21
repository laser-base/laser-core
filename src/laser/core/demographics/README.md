# `laser.core.demographics`

Population-pyramid and survival-curve utilities used to initialize, age, and
attrit agent populations in LASER simulations.

## Modules

| Module | What it provides |
| --- | --- |
| `pyramid.py` | `AliasedDistribution` (Vose alias sampler) and `load_pyramid_csv` (loader for UN-format age pyramids). |
| `kmestimator.py` | `KaplanMeierEstimator` — predict the year or day of death from a cumulative-deaths source. |
| `spatialpops.py` | Generate plausible per-node populations for spatial scenarios. |

## Reference documentation

- `docs/pyramids.rst` — using population pyramids for initialization.
- `docs/kmestimator.rst` — Kaplan-Meier estimator reference and example.
- `docs/vdexample.rst` — vital-dynamics worked example.

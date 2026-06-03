# `examples/` — runnable demos and input data

Jupyter notebooks and supporting CSVs that exercise core `laser-core` workflows.

## Notebooks

| Notebook | What it demonstrates |
| --- | --- |
| `age_pyramid.ipynb` | Loading and visualizing a UN-format age pyramid; sampling from it with `AliasedDistribution`. |
| `kmestimator.ipynb` | Using the Kaplan-Meier estimator to predict year-of-death and day-of-death from a cumulative-deaths source. |

## Input data

| File | Description |
| --- | --- |
| `Nigeria-2024.csv` | UN World Population Prospects 2024 age pyramid for Nigeria. |
| `Nigeria-2024.png` | Rendered age-pyramid plot for Nigeria. |
| `United States of America-2024.csv` | UN World Population Prospects 2024 age pyramid for the USA. |
| `United States of America - 2024.png` | Rendered age-pyramid plot for the USA. |

## Related references

Longer-form, narrative examples live in `docs/`:

- `docs/example.rst` — SIR walkthrough using `LaserFrame`.
- `docs/vdexample.rst` — vital-dynamics (births and deaths) walkthrough.
- `docs/spatialexample.rst` — spatial / migration walkthrough.

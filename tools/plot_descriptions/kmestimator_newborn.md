### Reading the newborn-cohort plot

Both curves are **cumulative deaths by age**, plotted on twin y-axes that share the same 0–100,000 range so the lines should sit on top of each other if the sampler is correct.

- **Green line — input cumulative deaths** straight from the `cumulative` array supplied to `KaplanMeierEstimator` (Nigeria-like life table, 100k-person cohort).
- **Orange line with × markers — sampled cumulative deaths** computed by binning the 100,000 predicted dates of death by year and taking the running sum.

The S-curve has three regions: a sharp early-life jump to ~16,000 deaths by age 10 (infant/child mortality), a near-linear middle through working ages, and a steep climb between ages 60 and 90 that asymptotes at 100,000 by ~100. **The orange sampled curve overlays the green input curve essentially perfectly across the entire age range** — the Kaplan–Meier estimator reproduces the input life table when seeded with newborn agents.

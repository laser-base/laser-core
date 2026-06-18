### Reading the 50+ cohort plot

Same twin-axis cumulative-deaths layout as the newborn plot, but here the input population consists of agents whose dates of birth are fixed at 50 years old, so only deaths from age 50 onward should appear.

- **Green line — input cumulative deaths conditional on surviving to 50**, i.e. `cumulative - cumulative[49]` clipped at zero.
- **Orange line with × markers — sampled cumulative deaths** from `predict_age_at_death()` called on the ~62k agents expected to still be alive at age 50.

Both curves are flat at zero from age 0 through 49 (no deaths can occur before the cohort's start age), then rise as an S-curve from age 50 to ~100, levelling off near 62,000 — the size of the conditional cohort. **The orange and green curves overlap throughout the rising region**, demonstrating that the estimator correctly conditions on current age rather than re-drawing from the unconditional life table.

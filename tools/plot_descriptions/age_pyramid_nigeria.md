### Reading the Nigeria plot

The figure overlays two curves against the same age-bin axis (0–4 through 100+):

- **Green line — input distribution.** The male-population fraction in each 5-year bin, taken straight from the Nigeria 2024 pyramid.
- **Orange line with × markers — sampled distribution.** The empirical fraction of the 100,000 sampled agents that fell into each bin after `AliasedDistribution.sample()` followed by a uniform draw within each bin.

Both curves share the y-axis range 0–10%. The shape is the classic expansive pyramid: ~7.5% of the population in the youngest 0–4 bin, declining roughly monotonically to near-zero by age 80+. **The takeaway is that the orange sampled curve lies essentially on top of the green input curve across every bin** — visual confirmation that the alias sampler reproduces the source pyramid within Monte-Carlo noise at N=100k.

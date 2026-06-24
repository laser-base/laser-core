"""
Illustrate why calc_capacity() must compute the peak-across-time of the cumulative
net-growth exponent (births minus credited deaths) rather than the end-of-simulation
value. Fluctuating CBR/CDR can produce an intermediate peak in living population well
above the end-of-sim value; an end-of-sim bound would under-allocate the LaserFrame.

Scenario:
    - 100,000 initial population.
    - 20 years (7300 daily ticks), single node.
    - CBR linearly DECREASING from 40 -> 10 per 1,000 per year.
    - CDR linearly INCREASING from 5 -> 35 per 1,000 per year.

Outputs a PNG (next to this script) showing:
    - The deterministic population trajectory under those rates.
    - The current calc_capacity() peak-living estimate (matches the trajectory peak).
    - The naive end-of-sim estimate (under-allocates by ~10%).

Run with: .venv/bin/python3 misc/calc_capacity_peak_vs_end.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from laser.core.utils import calc_capacity

# ── Scenario ─────────────────────────────────────────────────────────────────────────
N_YEARS = 20
NTICKS = N_YEARS * 365
NNODES = 1
INITIAL_POP = 100_000

# Year-by-year linear ramps.
cbr_per_year = np.linspace(40.0, 10.0, N_YEARS)  # decreasing CBR
cdr_per_year = np.linspace(5.0, 35.0, N_YEARS)  # increasing CDR

# Broadcast year-values across daily ticks.
cbr = np.repeat(cbr_per_year, 365).reshape(NTICKS, NNODES).astype(np.float32)
cdr = np.repeat(cdr_per_year, 365).reshape(NTICKS, NNODES).astype(np.float32)
initial_pop = np.array([INITIAL_POP], dtype=np.int64)

# ── Deterministic trajectory ─────────────────────────────────────────────────────────
# Same daily-growth formula calc_capacity uses, so the trajectory and the estimate are
# directly comparable.
lamda_b = (1.0 + cbr / 1000) ** (1.0 / 365) - 1.0
lamda_d = (1.0 + cdr / 1000) ** (1.0 / 365) - 1.0
net_daily = (lamda_b - lamda_d).flatten()

cumulative_net = np.cumsum(net_daily)
# pop(t=0) == INITIAL_POP; pop(t>0) = INITIAL_POP * exp(sum_{i<=t} net_daily[i])
pop_t = np.empty(NTICKS + 1)
pop_t[0] = INITIAL_POP
pop_t[1:] = INITIAL_POP * np.exp(cumulative_net)
times = np.arange(NTICKS + 1) / 365.0  # years (0..20 inclusive)

# ── Estimates ────────────────────────────────────────────────────────────────────────
# Tight estimate (no headroom): safety_factor=0, mortality_safety_factor=0 — credits all
# deaths and adds no births variance. Apples-to-apples vs the deterministic trajectory.
estimate_tight = int(calc_capacity(cbr, initial_pop, safety_factor=0.0, deathrates=cdr, mortality_safety_factor=0.0)[0])

# Default estimate (matches what library users get by default): safety_factor=1.0,
# mortality_safety_factor=1.0 (credits half the deaths, adds sqrt-shaped births headroom).
estimate_default = int(calc_capacity(cbr, initial_pop, safety_factor=1.0, deathrates=cdr, mortality_safety_factor=1.0)[0])

# Naive end-of-sim formula (what calc_capacity used to compute pre-fix). Hand-computed
# here for direct comparison: floor(initial_pop * exp(sum_b - sum_d)).
end_of_sim_estimate = int(round(INITIAL_POP * float(np.exp(net_daily.sum()))))
end_of_sim_estimate = max(end_of_sim_estimate, INITIAL_POP)

# Trajectory peak (the truth we're trying to bound).
peak_idx = int(np.argmax(pop_t))
peak_value = int(round(pop_t[peak_idx]))
peak_year = times[peak_idx]

# ── Plot ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(times, pop_t, color="C0", linewidth=2.0, label="Deterministic population trajectory")

ax.axhline(INITIAL_POP, color="grey", linestyle=":", linewidth=1.2, label=f"Initial pop = {INITIAL_POP:,}")
ax.axhline(
    estimate_tight,
    color="C2",
    linestyle="--",
    linewidth=1.4,
    label=f"calc_capacity peak-living (tightest) = {estimate_tight:,}",
)
ax.axhline(
    estimate_default,
    color="C3",
    linestyle="--",
    linewidth=1.4,
    label=f"calc_capacity peak-living (default sf=msf=1) = {estimate_default:,}",
)
ax.axhline(
    end_of_sim_estimate,
    color="C1",
    linestyle="--",
    linewidth=1.4,
    label=f"Naive end-of-sim (BUGGY) = {end_of_sim_estimate:,} — under-allocates",
)

ax.plot(
    peak_year,
    peak_value,
    "o",
    color="C0",
    markersize=8,
    label=f"Trajectory peak ≈ {peak_value:,} at year {peak_year:.1f}",
)

ax.set_xlabel("Years")
ax.set_ylabel("Population")
ax.set_title("Why calc_capacity() needs peak-across-time, not end-of-sim\n" f"{INITIAL_POP:,} initial, CBR 40→10, CDR 5→35 over {N_YEARS}y")
ax.set_xlim(0, N_YEARS)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=9)

plt.tight_layout()
out = Path(__file__).parent / "calc_capacity_peak_vs_end.png"
plt.savefig(out, dpi=120)
print(f"trajectory peak  : {peak_value:>10,} at year {peak_year:.2f}")
print(f"peak-living tight: {estimate_tight:>10,}  ({estimate_tight / peak_value:.4f} x peak)")
print(f"peak-living dflt : {estimate_default:>10,}  ({estimate_default / peak_value:.4f} x peak)")
print(f"end-of-sim bug   : {end_of_sim_estimate:>10,}  ({end_of_sim_estimate / peak_value:.4f} x peak — UNDER)")
print(f"wrote {out}")

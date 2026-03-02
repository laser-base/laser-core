"""
test_snapshot_continuity.py

Model-agnostic snapshot continuity test for LaserFrame.

Builds a minimal synthetic ABM with deterministic vital dynamics
(fixed lifespan + constant birth rate) and runs it in two ways:

  - Flat:   a single uninterrupted run for TOTAL_STEPS steps
  - Staged: run SNAP_STEP steps, save snapshot, reload, continue to TOTAL_STEPS

The deterministic design (fixed lifespan, no stochastic variation) makes
correctness easy to reason about without per-step PRNG alignment.

Assertions verify the three concrete bugs documented in polio_snapshot_for_core.md:
  1. date_of_death offset   — no agent dies at step 0 or earlier after reload
  2. pop_final continuity   — population at start of seg2 == end of seg1
  3. no death-rate spike    — deaths/step in first FIXED_LIFESPAN steps of seg2
                              are within 50% of seg1's mean rate

Run as a script for a visual boundary check:
    python tests/test_snapshot_continuity.py
"""

import tempfile
from pathlib import Path

import numpy as np

from laser.core import LaserFrame

# ── Simulation constants ───────────────────────────────────────────────────────
INIT_POP = 2_000
FIXED_LIFESPAN = 40  # every agent lives exactly this many steps
BIRTH_RATE = 1.0 / FIXED_LIFESPAN  # births/person/step — holds population steady
TOTAL_STEPS = 160
SNAP_STEP = 80  # save snapshot here, midway through


# ── Minimal ABM helpers ────────────────────────────────────────────────────────


def _make_frame(capacity: int, count: int, t_offset: int) -> LaserFrame:
    """Create a LaserFrame with date_of_death set relative to t_offset."""
    frame = LaserFrame(capacity=capacity, initial_count=count)
    frame.add_scalar_property("date_of_death", dtype=np.int32, default=0)
    # All agents born at t_offset; they die exactly FIXED_LIFESPAN steps later.
    frame.date_of_death[:] = t_offset + FIXED_LIFESPAN
    return frame


def _run_segment(frame: LaserFrame, n_steps: int, t_start: int) -> np.ndarray:
    """
    Advance *frame* for n_steps and return population time-series of shape (n_steps+1,).

    t_start is the absolute timestep corresponding to frame's current state (step 0
    of this segment).  Deaths fire when date_of_death <= current absolute step.
    """
    pop_ts = np.zeros(n_steps + 1, dtype=np.int64)
    pop_ts[0] = frame.count

    for step in range(1, n_steps + 1):
        t_abs = t_start + step

        # Deaths: remove agents whose absolute death time has arrived
        alive = frame.date_of_death[: frame.count] > t_abs
        frame.squash(alive)

        # Births: keep population near steady state
        n_births = max(0, round(frame.count * BIRTH_RATE))
        n_births = min(n_births, frame.capacity - frame.count)
        if n_births > 0:
            start, end = frame.add(n_births)
            frame._date_of_death[start:end] = t_abs + FIXED_LIFESPAN

        pop_ts[step] = frame.count

    return pop_ts


# ── Flat run (ground truth) ────────────────────────────────────────────────────


def _flat_run() -> tuple[np.ndarray, LaserFrame]:
    capacity = int(INIT_POP * 2)
    frame = _make_frame(capacity, INIT_POP, t_offset=0)
    pop_ts = _run_segment(frame, TOTAL_STEPS, t_start=0)
    return pop_ts, frame


# ── Staged run (snapshot + reload) ────────────────────────────────────────────


def _staged_run(snap_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (pop_seg1, pop_seg2) where each is shape (n_steps+1,).
    pop_seg1 covers steps 0..SNAP_STEP; pop_seg2 covers steps 0..TOTAL_STEPS-SNAP_STEP.
    """
    capacity = int(INIT_POP * 2)

    # --- Segment 1 ---
    frame1 = _make_frame(capacity, INIT_POP, t_offset=0)
    pop_seg1 = _run_segment(frame1, SNAP_STEP, t_start=0)

    pop_final = np.array([frame1.count], dtype=np.int64)
    frame1.save_snapshot(snap_path, t=SNAP_STEP, pop_final=pop_final)

    # --- Segment 2: reload from snapshot ---
    loaded, _, pars = LaserFrame.load_snapshot(snap_path, cbr=None, nt=None)

    # Rebuild with enough capacity for births in seg2
    seg2_steps = TOTAL_STEPS - SNAP_STEP
    capacity2 = int(loaded.count * 2)
    frame2 = LaserFrame(capacity=capacity2, initial_count=loaded.count)
    frame2.add_scalar_property("date_of_death", dtype=np.int32, default=0)
    # date_of_death has already been offset to seg2's timeline by load_snapshot
    frame2.date_of_death[:] = loaded.date_of_death[:]

    pop_seg2 = _run_segment(frame2, seg2_steps, t_start=0)

    # Restore pop[0] of seg2 from pop_final so the stitched series is continuous
    if "pop_final" in pars:
        pop_seg2[0] = int(pars["pop_final"].sum())

    return pop_seg1, pop_seg2


# ── pytest tests ──────────────────────────────────────────────────────────────


def test_date_of_death_all_positive_after_reload():
    """
    After reload, every agent's date_of_death is >= 1.
    Without the t_snap offset fix, agents whose absolute death time is <= t_snap
    would have date_of_death <= 0, causing immediate mass death at step 1.
    """
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        snap_path = tmp.name
    try:
        capacity = int(INIT_POP * 2)
        frame1 = _make_frame(capacity, INIT_POP, t_offset=0)
        _run_segment(frame1, SNAP_STEP, t_start=0)
        frame1.save_snapshot(snap_path, t=SNAP_STEP)

        loaded, _, pars = LaserFrame.load_snapshot(snap_path, cbr=None, nt=None)

        assert pars["t_snap"] == SNAP_STEP
        assert loaded.count > 0
        assert np.all(loaded.date_of_death > 0), (
            f"Some agents have date_of_death <= 0 after reload. " f"Min value: {loaded.date_of_death.min()}"
        )
    finally:
        Path(snap_path).unlink(missing_ok=True)


def test_date_of_death_within_one_lifespan_after_reload():
    """
    After reload all date_of_death values are in [1, FIXED_LIFESPAN].
    Without the offset fix they'd be in (SNAP_STEP, SNAP_STEP + FIXED_LIFESPAN],
    causing agents to live SNAP_STEP steps too long in the new segment.
    """
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        snap_path = tmp.name
    try:
        capacity = int(INIT_POP * 2)
        frame1 = _make_frame(capacity, INIT_POP, t_offset=0)
        _run_segment(frame1, SNAP_STEP, t_start=0)
        frame1.save_snapshot(snap_path, t=SNAP_STEP)

        loaded, _, _ = LaserFrame.load_snapshot(snap_path, cbr=None, nt=None)

        # Agents born throughout seg1 have lifetimes in [1, FIXED_LIFESPAN]
        assert np.all(loaded.date_of_death >= 1)
        assert np.all(loaded.date_of_death <= FIXED_LIFESPAN), (
            f"date_of_death exceeds FIXED_LIFESPAN ({FIXED_LIFESPAN}). "
            f"Max value: {loaded.date_of_death.max()} — t_snap offset was not applied."
        )
    finally:
        Path(snap_path).unlink(missing_ok=True)


def test_pop_final_continuity():
    """
    Population at the start of seg2 (pop_seg2[0]) equals population at the end
    of seg1 (pop_seg1[-1]) when pop_final is used.
    """
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        snap_path = tmp.name
    try:
        pop_seg1, pop_seg2 = _staged_run(snap_path)
        assert pop_seg2[0] == pop_seg1[-1], (
            f"Population discontinuity at boundary: " f"end of seg1={pop_seg1[-1]}, start of seg2={pop_seg2[0]}"
        )
    finally:
        Path(snap_path).unlink(missing_ok=True)


def test_no_death_spike_at_boundary():
    """
    Deaths per step in the first FIXED_LIFESPAN steps of seg2 are within 50% of
    seg1's mean death rate.  Without the t_snap offset, no agents die early in
    seg2 (they're all scheduled too far in the future), then all die at once.
    """
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        snap_path = tmp.name
    try:
        pop_seg1, pop_seg2 = _staged_run(snap_path)

        # Deaths = population drop between consecutive steps (births add, so drops only)
        def deaths_per_step(pop_ts):
            diffs = np.diff(pop_ts.astype(np.int64))
            return np.maximum(-diffs, 0)  # ignore birth-driven increases

        d1 = deaths_per_step(pop_seg1)
        d2 = deaths_per_step(pop_seg2)

        mean_d1 = d1[1:].mean()  # skip step 0 (no deaths at t=0)
        # Look at death rate in the early part of seg2 (within one lifespan)
        early_d2 = d2[1 : FIXED_LIFESPAN + 1].mean()

        assert early_d2 > 0, "No deaths at all in seg2 — t_snap offset fix was not applied"
        assert early_d2 < mean_d1 * 1.5, f"Death rate spike at boundary: seg1 mean={mean_d1:.1f}, " f"seg2 early mean={early_d2:.1f}"
    finally:
        Path(snap_path).unlink(missing_ok=True)


# ── Visual boundary check (run as script) ─────────────────────────────────────


def _boundary_check(pop_seg1: np.ndarray, pop_seg2: np.ndarray) -> None:
    """Print a table of population values around the snapshot boundary."""
    # Stitch: seg1[0..SNAP_STEP] + seg2[1..] (seg2[0] is a repeat of seg1[-1])
    stitched = np.concatenate([pop_seg1, pop_seg2[1:]])
    snap = SNAP_STEP

    print(f"\n── Boundary check (step {snap - 1} → {snap} → {snap + 1}) ──")
    print(f"  {'step':<6}  {'population':>12}  {'delta':>8}")
    print("  " + "-" * 32)
    for i in range(max(0, snap - 3), min(len(stitched), snap + 4)):
        delta = int(stitched[i]) - int(stitched[i - 1]) if i > 0 else 0
        marker = "  ← snapshot" if i == snap else ""
        print(f"  {i:<6}  {stitched[i]:>12,}  {delta:>+8,}{marker}")


if __name__ == "__main__":
    print(f"Flat run ({TOTAL_STEPS} steps, pop={INIT_POP}, lifespan={FIXED_LIFESPAN}) ...")
    pop_flat, _ = _flat_run()
    print(f"  Final count: {pop_flat[-1]:,}")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        snap_path = tmp.name

    print(f"\nStaged run (snap @ step {SNAP_STEP}) ...")
    pop_seg1, pop_seg2 = _staged_run(snap_path)
    print(f"  Seg1 final: {pop_seg1[-1]:,}   Seg2 start: {pop_seg2[0]:,}")

    _boundary_check(pop_seg1, pop_seg2)

    Path(snap_path).unlink(missing_ok=True)

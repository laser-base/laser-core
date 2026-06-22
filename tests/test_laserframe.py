"""
Unit tests for the LaserFrame class in the laser.core.laserframe module.

This module contains a series of unit tests for the LaserFrame class, which is
designed to manage a collection of agents with various properties. The tests
cover initialization, property addition, agent addition, sorting, and squashing
functionality.

Classes:
    TestLaserFrame: A unittest.TestCase subclass that contains tests for the
    LaserFrame class.

Test Methods:
    - test_init: Tests the initialization of a LaserFrame instance with a
      specified capacity.
    - test_init_with_properties: Tests the initialization of a LaserFrame
      instance with additional properties.
    - test_add_scalar_property: Tests the addition of a scalar property with a
      default value.
    - test_add_scalar_property_with_value: Tests the addition of a scalar
      property with a specified default value.
    - test_add_property: Tests the deprecated add_property method (should use
      add_scalar_property).
    - test_add_property_with_value: Tests the deprecated add_property method
      with a specified default value (should use add_scalar_property).
    - test_add_vector_property: Tests the addition of a vector property with a
      specified length.
    - test_add_agents: Tests the addition of agents to the LaserFrame.
    - test_add_agents_again: Tests the addition of agents to the LaserFrame
      multiple times.
    - test_sort: Tests the sorting of agents based on a scalar property.
    - test_squash: Tests the squashing (filtering) of agents based on a
      condition.

Usage:
    Run this module with a Python interpreter to execute the unit tests.
"""

import csv
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from laser.core import LaserFrame
from laser.core import PropertySet
from laser.core.utils import calc_capacity


@pytest.mark.parametrize("pop_size", [10_000, 100_000, 1_000_000, 10_000_000])
def test_loaded_capacity_matches_expected_growth_single_node(pop_size):
    """
    After saving and reloading a LaserFrame snapshot with a known population,
    2-D CBR, and duration, the reloaded frame should have sufficient capacity
    to model expected population growth.
    """
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = tmp.name

    try:
        # Simulation parameters
        nt = 365 * 3  # 3 years
        nnodes = 1
        cbr_value = 35.0

        # Strict 2-D CBR (time, node)
        cbr = np.full((nt, nnodes), cbr_value, dtype=np.float32)
        initial_pop = np.array([pop_size], dtype=np.int32)

        # Expected final population
        expected_final = calc_capacity(
            birthrates=cbr,
            initial_pop=initial_pop,
        ).sum()

        # Create frame
        frame = LaserFrame(capacity=int(expected_final), initial_count=pop_size)
        frame.add_scalar_property("age", dtype=np.int32)
        frame.add_scalar_property("status", dtype=np.int8)
        frame.add_scalar_property("node_id", dtype=np.int32, default=0)

        # Dummy data
        np.random.seed(42)
        frame.age[:] = np.random.randint(0, 90, size=pop_size)
        frame.status[:] = np.random.choice([0, 1], size=pop_size)
        # All agents in node 0 for single-node case
        frame.node_id[:] = 0

        # Squash some agents to simulate churn
        mask = (frame.status == 1) | (frame.age > 75)
        frame.squash(~mask)

        # Save snapshot with non-zero recovered at t=0
        results_r = np.zeros((nt, nnodes), dtype=np.float32)
        results_r[0, 0] = 100.0
        frame.save_snapshot(path, results_r=results_r, pars={})

        # Reload snapshot
        loaded, _, _ = LaserFrame.load_snapshot(path, cbr=cbr, nt=nt)

        # Recompute expected capacity using *modeled* agents only
        expected_capacity = calc_capacity(
            birthrates=cbr,
            initial_pop=np.array([frame.count], dtype=np.int32),
        ).sum()

        # Assertions
        assert loaded.count == frame.count
        assert loaded.capacity >= expected_capacity
        assert loaded.capacity <= expected_capacity * 1.1

    finally:
        Path(path).unlink()


@pytest.mark.parametrize("nnodes", [1, 3])
def test_loaded_capacity_matches_expected_growth_multi_node(nnodes):
    """
    Verify that load_snapshot correctly recomputes capacity when given
    a strict 2-D CBR array (time, node) in a multi-node scenario.

    Since snapshots do not store per-node population, load_snapshot
    aggregates nodes internally and must allocate sufficient capacity
    for total population growth.
    """
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = tmp.name

    try:
        nt = 365 * 3
        cbr_value = 35.0
        pop_size = 100_000

        # Distribute population evenly across nodes
        ppl = np.array([pop_size // nnodes] * nnodes, dtype=np.int32)

        # Strict 2-D CBR
        cbr = np.full((nt, nnodes), cbr_value, dtype=np.float32)

        # Expected total population growth (true multi-node math)
        expected_final = calc_capacity(
            birthrates=cbr,
            initial_pop=ppl,
        ).sum()

        # Create frame
        frame = LaserFrame(capacity=int(expected_final), initial_count=pop_size)
        frame.add_scalar_property("age", dtype=np.int32)
        frame.add_scalar_property("status", dtype=np.int8)
        frame.add_scalar_property("node_id", dtype=np.int32, default=0)

        # Dummy data
        np.random.seed(42)
        frame.age[:] = np.random.randint(0, 90, size=pop_size)
        frame.status[:] = np.random.choice([0, 1], size=pop_size)
        # Distribute agents across nodes (round-robin)
        frame.node_id[:] = np.arange(pop_size) % nnodes

        # Squash
        mask = (frame.status == 1) | (frame.age > 75)
        frame.squash(~mask)

        # Save snapshot with recovered counts per node at t=0
        results_r = np.zeros((nt, nnodes), dtype=np.float32)
        results_r[0, :] = 50.0
        frame.save_snapshot(path, results_r=results_r, pars={})

        # Reload snapshot
        loaded, _, _ = LaserFrame.load_snapshot(path, cbr=cbr, nt=nt)

        # load_snapshot aggregates nodes → treat as single population
        aggregated_cbr = np.mean(cbr, axis=1, keepdims=True)
        expected_capacity = calc_capacity(
            birthrates=aggregated_cbr,
            initial_pop=np.array([frame.count], dtype=np.int32),
        ).sum()

        # Assertions
        assert loaded.count == frame.count
        assert loaded.capacity >= expected_capacity
        assert loaded.capacity <= expected_capacity * 1.1

    finally:
        Path(path).unlink()


def _load_un_member_states_2020():
    """Read tests/data/un_member_states_2020.csv into parallel name/population lists."""
    csv_path = Path(__file__).parent / "data" / "un_member_states_2020.csv"
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        rows = [(name, int(pop)) for name, pop in reader]
    names = [r[0] for r in rows]
    pops = np.array([r[1] for r in rows], dtype=np.int64)
    return names, pops


def _available_memory_bytes():
    """Best-effort lookup of available RAM. Returns None if it cannot be determined."""
    try:
        import psutil  # — optional dep, lazily imported only for gating
    except ImportError:
        return None
    return psutil.virtual_memory().available


# Memory budget: capacity * (sizeof(uint16 age) + sizeof(uint8 state)) ≈ 3 bytes/agent.
# For ~7.84e9 agents that's ~23.5 GB just for the property arrays. We require a
# comfortable headroom for Python/NumPy overhead and the OS — skip below ~25 GB free.
_UN_2020_REQUIRED_FREE_BYTES = 25 * 1024**3


@pytest.mark.skipif(
    (_available_memory_bytes() or 0) < _UN_2020_REQUIRED_FREE_BYTES,
    reason=(
        f"Test requires ~{_UN_2020_REQUIRED_FREE_BYTES / 1024**3:.0f} GiB free RAM to allocate a LaserFrame "
        "with more than uint32.max agents (age uint16 + state uint8 across all UN 2020 populations)."
    ),
)
def test_un_member_states_2020_laserframe_exceeds_uint32_max():
    """Build a LaserFrame sized for the combined 2020 populations of all 193 UN member states.

    Given:
        The 2020 UN population estimates for all 193 UN member states (one node per country),
        whose sum exceeds ``np.iinfo(np.uint32).max`` (≈ 4.29 × 10^9 agents).
    When:
        We construct a ``LaserFrame`` with ``capacity`` equal to that total and add the
        bare-bones per-agent properties ``age`` (``uint16``) and ``state`` (``uint8``).
    Then:
        - The frame's ``count`` and ``capacity`` both exceed ``uint32.max``.
        - The ``age`` and ``state`` property arrays have the expected dtypes and span the
          full active range (shape ``(count,)``).
        - Indexing across the uint32 boundary works (we read and write a single agent
          located at index ``uint32.max + 1`` to confirm the underlying NumPy storage uses
          64-bit indexing under the hood).

    Failure of this test indicates LaserFrame cannot safely handle agent counts above
    ``2**32 - 1`` — a regression that would break global-scale or large-region
    simulations (e.g. continent-wide or world-wide scenarios).

    Note:
        This test allocates ~23-24 GiB of NumPy arrays and is skipped automatically on
        machines without sufficient free RAM (see ``_UN_2020_REQUIRED_FREE_BYTES``).
    """
    names, pops = _load_un_member_states_2020()

    assert len(names) == 193, f"Expected 193 UN member states, got {len(names)}"
    assert len(set(names)) == 193, "UN member state names must be unique"

    uint32_max = int(np.iinfo(np.uint32).max)
    total_pop = int(pops.sum())
    assert total_pop > uint32_max, (
        f"Combined 2020 population ({total_pop:_}) should exceed uint32 max ({uint32_max:_}); " "the test is moot if it does not."
    )

    frame = LaserFrame(capacity=total_pop, initial_count=total_pop)
    frame.add_scalar_property("age", dtype=np.uint16, default=0)
    frame.add_scalar_property("state", dtype=np.uint8, default=0)

    assert frame.capacity == total_pop
    assert frame.count == total_pop
    assert frame.count > uint32_max, "Frame count must exceed uint32 max for this test to be meaningful."
    assert frame.age.dtype == np.uint16, f"Expected uint16 age dtype, got {frame.age.dtype}"
    assert frame.state.dtype == np.uint8, f"Expected uint8 state dtype, got {frame.state.dtype}"
    # LaserFrame returns properties sliced to [0:count]; with count == capacity that's the full backing array.
    assert frame.age.shape == (total_pop,)
    assert frame.state.shape == (total_pop,)

    # Verify 64-bit indexing across the uint32 boundary — write/read at index uint32_max + 1
    # to confirm we are NOT silently wrapping into the lower 4.29 G of the array.
    sentinel_index = uint32_max + 13
    assert sentinel_index < total_pop, "Sanity: sentinel index must lie inside the frame."

    frame.age[sentinel_index] = 42
    frame.state[sentinel_index] = 7
    assert int(frame.age[sentinel_index]) == 42
    assert int(frame.state[sentinel_index]) == 7

    # The corresponding index in the LOWER half should NOT have been touched by the write.
    lower_index = sentinel_index - 2**32  # would be 12 if uint32 wrap occurred
    assert int(frame.age[lower_index]) == 0, "Write at uint32_max+13 must not wrap into the lower half (uint32 overflow)."
    assert int(frame.state[lower_index]) == 0, "Write at uint32_max+13 must not wrap into the lower half (uint32 overflow)."


class TestLaserFrame(unittest.TestCase):
    def test_init(self):
        pop = LaserFrame(1024, initial_count=0)
        assert pop.capacity == 1024
        assert pop.count == 0
        assert len(pop) == pop.count

        return

    def test_init_with_count(self):
        pop = LaserFrame(1024, initial_count=128)
        assert pop.capacity == 1024
        assert pop.count == 128
        assert len(pop) == pop.count

        return

    def test_init_with_count_minus_one(self):
        pop = LaserFrame(1024, initial_count=-1)
        assert pop.capacity == 1024
        assert pop.count == 1024
        assert len(pop) == pop.count

        return

    def test_init_with_properties(self):
        pop = LaserFrame(1024, initial_count=0, start_year=1944, source="https://ourworldindata.org/grapher/life-expectancy?country=~USA")
        assert pop.capacity == 1024
        assert pop.count == 0
        assert len(pop) == pop.count
        assert pop.start_year == 1944
        assert pop.source == "https://ourworldindata.org/grapher/life-expectancy?country=~USA"

        return

    def test_add_scalar_property(self):
        pop = LaserFrame(1024)
        pop.add_scalar_property("age", default=0)
        assert np.all(pop.age == 0)
        assert pop.age.shape == (1024,)

        return

    def test_add_scalar_property_with_value(self):
        pop = LaserFrame(1024)
        pop.add_scalar_property("age", default=10)
        assert np.all(pop.age == 10)
        assert pop.age.shape == (1024,)

        return

    def test_add_vector_property(self):
        pop = LaserFrame(1024)
        pop.add_vector_property("events", 365)
        assert np.all(pop.events == 0)
        assert pop.events.shape == (365, 1024)

        return

    def test_add_vectory_property_with_value(self):
        pop = LaserFrame(1024)
        pop.add_vector_property("events", 365, default=1)
        assert np.all(pop.events == 1)
        assert pop.events.shape == (365, 1024)

        return

    def test_add_array_property(self):
        pop = LaserFrame(1024)
        pop.add_array_property("events", (365, 1024))
        assert np.all(pop.events == 0)
        assert pop.events.shape == (365, 1024)

        return

    def test_add_array_property_with_value(self):
        pop = LaserFrame(1024)
        pop.add_array_property("events", (365, 1024), default=42)
        assert np.all(pop.events == 42)
        assert pop.events.shape == (365, 1024)

        return

    def test_add_array_property_with_dtype(self):
        pop = LaserFrame(1024)
        default = np.float32(-3.14159265)
        pop.add_array_property("events", (365, 1024), dtype=np.float32, default=default)
        assert np.all(pop.events == default)
        assert pop.events.shape == (365, 1024)
        assert pop.events.dtype == np.float32

        return

    def test_add_agents(self):
        pop = LaserFrame(1024, 100)
        assert pop.count == 100
        assert len(pop) == pop.count
        istart, iend = pop.add(200)
        assert istart == 100
        assert iend == 300
        assert pop.count == 300
        assert len(pop) == pop.count

        return

    def test_add_agents_again(self):
        pop = LaserFrame(1024, 100)
        istart, iend = pop.add(200)
        istart, iend = pop.add(500)
        assert istart == 300
        assert iend == 800
        assert pop.count == 800
        assert len(pop) == pop.count

        return

    def test_add_too_many_agents(self):
        pop = LaserFrame(1024, 1000)
        assert pop.count == 1000
        assert len(pop) == pop.count

        with pytest.raises(
            ValueError, match=re.escape("frame.add() exceeds capacity (self._count=1000 + count=100 > self._capacity=1024)")
        ):
            pop.add(100)

        return

    def test_sort(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = pop.count
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)  # random ages 0-100 years
        original_age = np.array(pop.age)
        pop.height[istart:iend] = np.random.default_rng().uniform(0.5, 2.0, 100)  # random heights 0.5-2 meters
        original_height = np.array(pop.height)
        indices = np.argsort(pop.age)
        pop.sort(indices, verbose=False)
        assert np.all(pop.age == np.sort(original_age))
        assert np.all(pop.height == original_height[indices])

        return

    def test_sort_sanity_check(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = pop.count
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)
        indices = np.argsort(pop.age)

        with pytest.raises(TypeError, match=re.escape(f"Indices must be a numpy array (got {list})")):
            pop.sort(indices.tolist(), verbose=True)

        with pytest.raises(
            TypeError, match=re.escape(f"Indices must have the same length as the frame active element count ({pop.count})")
        ):
            pop.sort(indices[0:50], verbose=True)

        with pytest.raises(TypeError, match=re.escape("Indices must be an integer array (got float32)")):
            pop.sort(indices.astype(np.float32), verbose=1)

        return

    def test_squash(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = pop.count
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)  # random ages 0-100 years
        original_age = np.array(pop.age)
        pop.height[istart:iend] = np.random.default_rng().uniform(0.5, 2.0, 100)  # random heights 0.5-2 meters
        original_height = np.array(pop.height)
        keep = pop.age >= 40
        pop.squash(keep, verbose=False)
        assert pop.count == keep.sum()
        assert np.all(pop.age == original_age[keep])
        assert np.all(pop.height == original_height[keep])

        return

    def test_squash_sanity_checks(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = 100
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)
        keep = pop.age >= 40

        with pytest.raises(TypeError, match=re.escape(f"Indices must be a numpy array (got {list})")):
            pop.squash(keep.tolist(), verbose=True)

        with pytest.raises(
            TypeError, match=re.escape(f"Indices must have the same length as the frame active element count ({pop.count})")
        ):
            pop.squash(keep[0:50], verbose=True)

        with pytest.raises(TypeError, match=re.escape("Indices must be a boolean array (got float32)")):
            pop.squash(keep.astype(np.float32), verbose=1)

        return

    def test_init_bad_capacity1(self):
        capacity = "5150"
        with pytest.raises(ValueError, match=re.escape(f"Capacity must be a positive integer, got {capacity}.")):
            _ = LaserFrame(capacity=capacity)

        return

    def test_init_bad_capacity2(self):
        capacity = -5150
        with pytest.raises(ValueError, match=re.escape(f"Capacity must be a positive integer, got {capacity}.")):
            _ = LaserFrame(capacity=capacity)

        return

    def test_init_bad_initial_count1(self):
        initial_count = "5150"
        with pytest.raises(ValueError, match=re.escape(f"Initial count must be a non-negative integer, got {initial_count}.")):
            _ = LaserFrame(capacity=65536, initial_count=initial_count)

        return

    def test_init_bad_initial_count2(self):
        initial_count = -5150
        with pytest.raises(ValueError, match=re.escape(f"Initial count must be a non-negative integer, got {initial_count}.")):
            _ = LaserFrame(capacity=65536, initial_count=initial_count)

        return

    def test_init_bad_initial_count3(self):
        capacity = 65536
        initial_count = 1_000_000
        with pytest.raises(ValueError, match=re.escape(f"Initial count ({initial_count}) cannot exceed capacity ({capacity}).")):
            _ = LaserFrame(capacity=capacity, initial_count=initial_count)

        return

    def test_save_and_load_snapshot(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name

        try:
            # Create frame
            count = 10_000
            frame = LaserFrame(capacity=100_000, initial_count=count)
            frame.add_scalar_property("age", dtype=np.int32)
            frame.add_scalar_property("status", dtype=np.int8)
            frame.add_scalar_property("node_id", dtype=np.int32, default=0)

            # Assign values
            np.random.seed(42)
            frame.age[:] = np.random.randint(0, 100, size=count)
            frame.status[:] = np.random.choice([0, 1], size=count)  # 1 = recovered
            frame.node_id[:] = 0  # Single node

            # Squash agents who are recovered or age > 70
            mask = (frame.status == 1) | (frame.age > 70)
            removed = mask.sum()
            frame.squash(~mask)

            # Create a 1x10 time series of declining recovered counts
            results_r = np.linspace(removed, 0, 10, dtype=np.float32).reshape(1, -1)

            # Parameters
            pars = PropertySet({"r0": 2.5, "intervention": "vaccine"})

            # Save
            frame.save_snapshot(path, results_r=results_r, pars=pars)

            # Load
            loaded, r_loaded, pars_loaded = frame.load_snapshot(path, cbr=None, nt=None)

            assert loaded.count == frame.count
            assert np.array_equal(loaded.age, frame.age)
            assert np.array_equal(loaded.status, frame.status)
            assert np.array_equal(loaded.node_id, frame.node_id)
            assert np.array_equal(r_loaded, results_r)
            assert pars_loaded["r0"] == 2.5
            # print(f"pars_loaded={pars_loaded}")
            assert pars_loaded["intervention"] == "vaccine"

            # print("test_save_and_load_snapshot passed.")
        finally:
            Path(path).unlink()

        return

    def test_numpy_ints_for_capacity(self):
        for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            capacity = t(min(np.iinfo(t).max // 2, 1 << 10))  # Ensure capacity is reasonable
            lf = LaserFrame(capacity)  # just use default initial count
            assert lf.capacity == int(capacity), f"Expected capacity {int(capacity)}, got {lf.capacity}"
            assert lf.count == int(capacity), f"Expected count {int(capacity)}, got {lf.count}"

        return

    def test_numpy_ints_for_initial_count(self):
        for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            capacity = int(min(np.iinfo(t).max, 1 << 10))  # Ensure capacity is reasonable
            count = t(capacity // 2)
            lf = LaserFrame(capacity, initial_count=count)
            assert lf.capacity == int(capacity), f"Expected capacity {int(capacity)}, got {lf.capacity}"
            assert lf.count == int(count), f"Expected count {int(count)}, got {lf.count}"

        return

    def test_describe(self):
        # We use this rather than a triple-quoted string to avoid issues with
        # leading/trailing whitespace and newlines in the expected output.
        expected = "\n".join(
            [
                "",
                "Laserframe Report for `TestFrame`:",
                "Capacity:         1,024",
                "Count:              768",
                "",
                "=================================================================================================",
                "                                             Scalars                                             ",
                "=================================================================================================",
                "Name        | Datatype  | Individual Size (bytes) | Allocated Size (bytes) |  In Use Size (bytes)",
                "-------------------------------------------------------------------------------------------------",
                "temperature |  float32  |            4            |                  4,096 |                3,072",
                "-------------------------------------------------------------------------------------------------",
                "Total       |           |            4            |                  4,096 |                3,072",
                "-------------------------------------------------------------------------------------------------",
                "",
                "=========================================================================================================",
                "                                                 Vectors                                                 ",
                "=========================================================================================================",
                "Name       | Datatype  | Count  | Individual Size (bytes) | Allocated Size (bytes) |  In Use Size (bytes)",
                "---------------------------------------------------------------------------------------------------------",
                "velocities |  float32  |   3    |           12            |                 12,288 |                9,216",
                "---------------------------------------------------------------------------------------------------------",
                "Total      |           |        |           12            |                 12,288 |                9,216",
                "---------------------------------------------------------------------------------------------------------",
                "",
                "==========================================================================================================",
                "                                                  Others                                                  ",
                "==========================================================================================================",
                "Name        | Datatype | Individual Size (bytes) |      Shape      | Num Elements | Allocated Size (bytes)",
                "----------------------------------------------------------------------------------------------------------",
                "sensor_data | float64  |            8            |    (10, 10)     |     100      |                    800",
                "----------------------------------------------------------------------------------------------------------",
                "Total       |          |            8            |                 |              |                    800",
                "----------------------------------------------------------------------------------------------------------",
                "",
            ]
        )
        lf = LaserFrame(capacity=1024, initial_count=768)
        lf.add_scalar_property("temperature", dtype=np.float32)
        lf.add_vector_property("velocities", 3, dtype=np.float32)
        lf.add_array_property("sensor_data", shape=(10, 10), dtype=np.float64)

        assert lf.describe("TestFrame") == expected

        return

    def test_catch_duplicate_property(self):
        lf = LaserFrame(capacity=1024, initial_count=0)

        # Test add_scalar_property()
        lf.add_scalar_property("age", dtype=np.int32)

        with pytest.raises(ValueError, match=re.escape("Property 'age' already exists in LaserFrame.")):
            lf.add_scalar_property("age", dtype=np.int32)

        with pytest.raises(ValueError, match=re.escape("Property 'age' already exists in LaserFrame.")):
            lf.add_vector_property("age", 3, dtype=np.int32)

        with pytest.raises(ValueError, match=re.escape("Property 'age' already exists in LaserFrame.")):
            lf.add_array_property("age", shape=(10, 10), dtype=np.int32)

        # Test add_vector_property()
        lf.add_vector_property("position", 3, dtype=np.float32)

        with pytest.raises(ValueError, match=re.escape("Property 'position' already exists in LaserFrame.")):
            lf.add_scalar_property("position", dtype=np.float32)

        with pytest.raises(ValueError, match=re.escape("Property 'position' already exists in LaserFrame.")):
            lf.add_vector_property("position", 3, dtype=np.float32)

        with pytest.raises(ValueError, match=re.escape("Property 'position' already exists in LaserFrame.")):
            lf.add_array_property("position", shape=(10, 10), dtype=np.float32)

        # Test add_array_property()
        lf.add_array_property("sensor_data", shape=(10, 10), dtype=np.float32)

        with pytest.raises(ValueError, match=re.escape("Property 'sensor_data' already exists in LaserFrame.")):
            lf.add_scalar_property("sensor_data", dtype=np.float32)

        with pytest.raises(ValueError, match=re.escape("Property 'sensor_data' already exists in LaserFrame.")):
            lf.add_vector_property("sensor_data", 3, dtype=np.float32)

        with pytest.raises(ValueError, match=re.escape("Property 'sensor_data' already exists in LaserFrame.")):
            lf.add_array_property("sensor_data", shape=(10, 10), dtype=np.float32)

        return

    def test_underlying_array_access(self):
        lf = LaserFrame(capacity=1024, initial_count=0)
        lf.add_scalar_property("age", dtype=np.int32)
        lf.add_vector_property("position", 3, dtype=np.float32)

        # Access underlying arrays
        age_array = lf._age
        assert isinstance(age_array, np.ndarray)
        assert age_array.shape == (1024,)
        assert age_array.dtype == np.int32

        position_array = lf._position
        assert isinstance(position_array, np.ndarray)
        assert position_array.shape == (3, 1024)
        assert position_array.dtype == np.float32

        return

    # Test that accessing a property returns the correct slice, not the entire array
    # Let's create a LaserFrame with a given capacity and half that capacity as the initial count.
    # We'll also set the default value to something unique so we can easily identify it.
    # Then, we'll access the property and modify it, checking that only the active portion is affected.
    # This also tests that modifying the returned property modifies the underlying array correctly.
    # This also tests the condition where 0 < count < capacity.
    def test_property_slice_modification(self):
        capacity = 100
        initial_count = 50
        default_value = 7

        lf = LaserFrame(capacity=capacity, initial_count=initial_count)
        lf.add_scalar_property("test_prop", dtype=np.int32, default=default_value)

        # Access the property and modify it
        test_prop = lf.test_prop
        test_prop += 3  # Increment all values by 3

        # Check that only the active portion is modified
        assert np.all(lf._test_prop[:initial_count] == default_value + 3), "Active portion not modified correctly."
        assert np.all(lf._test_prop[initial_count:] == default_value), "Inactive portion should remain at default value."

        return

    # Test boundary conditions: count == 0
    def test_zero_count(self):
        lf = LaserFrame(capacity=100, initial_count=0)
        lf.add_scalar_property("age", dtype=np.int32, default=13)
        assert lf.count == 0
        assert len(lf) == 0
        assert lf._age.shape == (100,)
        assert lf.age.shape == (0,)

        return

    # Test boundary conditions: count == capacity
    def test_count_equals_capacity(self):
        lf = LaserFrame(capacity=100, initial_count=100)
        lf.add_scalar_property("age", dtype=np.int32, default=13)
        assert lf.count == 100
        assert len(lf) == 100
        assert lf._age.shape == (100,)
        assert np.all(lf._age == 13)
        assert lf.age.shape == (100,)
        assert np.all(lf.age == 13)
        lf.age[:] = 42
        assert np.all(lf.age == 42)
        assert np.all(lf._age == 42)

        return

    def test_cannot_reassign_property(self):
        lf = LaserFrame(capacity=100, initial_count=10)
        lf.add_scalar_property("age", dtype=np.int32, default=13)

        with pytest.raises(
            RuntimeError, match=re.escape("Cannot reassign property 'age'. Modify the array in place instead, e.g., lf.age[:] = new_values")
        ):
            lf.age = np.arange(10)

        return

    def test_dir_excludes_dynamic_property_names_on_empty_frame(self):
        """Given a fresh LaserFrame with no scalar/vector properties added,
        when ``dir(frame)`` is called,
        then names that will later be added (e.g. "age", "status") are NOT present, but
        the standard public surface (``count``, ``capacity``, ``add``, ``add_scalar_property``,
        ``sort``, ``squash``) IS present.

        Failure of this test means ``__dir__`` is leaking names that don't yet exist as
        attributes — REPL tab-completion would offer columns the user hasn't created,
        which is misleading.
        """
        lf = LaserFrame(capacity=128, initial_count=0)
        names = set(dir(lf))

        assert "age" not in names
        assert "status" not in names

        for expected in ("count", "capacity", "add", "add_scalar_property", "add_vector_property", "add_array_property", "sort", "squash"):
            assert expected in names, f"Expected {expected!r} in dir(LaserFrame), got {sorted(names)}"

        return

    def test_dir_includes_scalar_property_names(self):
        """Given a LaserFrame with scalar properties added via ``add_scalar_property``,
        when ``dir(frame)`` is called,
        then the public property names are present.

        This is what makes Jupyter/IPython tab-completion surface the dynamically-added
        columns. Without ``__dir__`` returning them, ``lf.<TAB>`` would miss the columns
        because they live in ``self._properties`` (under the underscored backing name in
        ``__dict__``) rather than as plain attributes.
        """
        lf = LaserFrame(capacity=128, initial_count=0)
        lf.add_scalar_property("age", dtype=np.int32)
        lf.add_scalar_property("status", dtype=np.int8)

        names = set(dir(lf))

        assert "age" in names, f"Expected 'age' in dir(lf), got {sorted(names)}"
        assert "status" in names, f"Expected 'status' in dir(lf), got {sorted(names)}"

        return

    def test_dir_includes_vector_property_names(self):
        """Given a LaserFrame with a vector property added via ``add_vector_property``,
        when ``dir(frame)`` is called,
        then the public property name is present.

        Vector properties are stored the same way as scalar properties (in
        ``self._properties`` with an underscored backing array), so they rely on the same
        ``__dir__`` merge to be discoverable.
        """
        lf = LaserFrame(capacity=64, initial_count=0)
        lf.add_vector_property("position", length=3, dtype=np.float32)

        names = set(dir(lf))

        assert "position" in names, f"Expected 'position' in dir(lf), got {sorted(names)}"

        return

    def test_dir_includes_array_property_names(self):
        """Given a LaserFrame with an array property added via ``add_array_property``,
        when ``dir(frame)`` is called,
        then the public property name is present.

        Unlike scalar/vector properties, ``add_array_property`` stores the array directly
        as an attribute (it lives in ``self.__dict__``), so it appears in dir() via the
        base class implementation — not via ``self._properties``. This test guards against
        any future change that moves array storage but forgets to update ``__dir__``.
        """
        lf = LaserFrame(capacity=32, initial_count=0)
        lf.add_array_property("sensor_data", shape=(10, 10), dtype=np.float32)

        names = set(dir(lf))

        assert "sensor_data" in names, f"Expected 'sensor_data' in dir(lf), got {sorted(names)}"

        return

    def test_dir_includes_kwargs_attributes(self):
        """Given a LaserFrame constructed with extra kwargs (e.g. ``start_year=1944``),
        when ``dir(frame)`` is called,
        then the kwarg names appear.

        Kwargs are stored as plain attributes (via ``setattr`` in ``__init__``), so they
        flow in through the base class's ``__dir__``. This test pins that behavior so a
        refactor that disabled kwarg setting would be caught.
        """
        lf = LaserFrame(capacity=128, initial_count=0, start_year=1944, source="ourworldindata")

        names = set(dir(lf))

        assert "start_year" in names, f"Expected 'start_year' in dir(lf), got {sorted(names)}"
        assert "source" in names, f"Expected 'source' in dir(lf), got {sorted(names)}"

        return

    def test_dir_is_sorted_and_unique(self):
        """Given a LaserFrame with several properties added,
        when ``dir(frame)`` is called,
        then the result is sorted in ascending order and contains no duplicates.

        ``__dir__`` is expected by convention (and by ``inspect``/``help``) to return a
        sorted list. The current implementation uses ``sorted(set(...) | set(...))`` —
        this test enforces both invariants.
        """
        lf = LaserFrame(capacity=64, initial_count=0)
        lf.add_scalar_property("age", dtype=np.int32)
        lf.add_scalar_property("status", dtype=np.int8)
        lf.add_vector_property("position", length=3, dtype=np.float32)
        lf.add_array_property("sensor_data", shape=(4, 4), dtype=np.float32)

        listing = dir(lf)

        assert listing == sorted(listing), "dir(LaserFrame) must return a sorted list"
        assert len(listing) == len(set(listing)), f"dir(LaserFrame) must not contain duplicates: {listing}"

        return

    def test_dir_grows_when_properties_are_added(self):
        """Given a LaserFrame, when properties are added,
        then ``dir(frame)`` grows by exactly the new public names (no spurious additions).

        Pins the contract that ``add_*_property`` is the ONLY way new names enter dir()
        through the dynamic path. Catches regressions where, say, the underscored
        backing name ``_age`` is double-added or auxiliary state leaks into the listing.
        """
        lf = LaserFrame(capacity=64, initial_count=0)
        before = set(dir(lf))

        lf.add_scalar_property("age", dtype=np.int32)
        lf.add_vector_property("position", length=3, dtype=np.float32)

        after = set(dir(lf))

        new = after - before
        # The backing arrays `_age` and `_position` also become attributes (via setattr),
        # so they too show up in dir() via the base class — that's expected and part of
        # the documented "underlying array access" surface (see test_underlying_array_access).
        assert new == {"age", "_age", "position", "_position"}, f"Unexpected additions to dir(LaserFrame): {sorted(new)}"

        return

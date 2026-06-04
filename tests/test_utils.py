import csv
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from laser.core.utils import calc_capacity
from laser.core.utils import grid
from laser.core.utils import initialize_population

City = namedtuple("City", ["name", "pop", "lat", "long"])


class TestUtilityFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def transmogrify(row):
            name_col = 0
            pop_col = 2
            lat_col = 9
            long_col = 10
            # [:-2] removes the degree symbol and "N" or "W"
            # -long because all US cities are West
            return City(
                name=row[name_col], pop=int(row[pop_col].replace(",", "")), lat=float(row[lat_col][:-2]), long=-float(row[long_col][:-2])
            )

        cities = Path(__file__).parent.absolute() / "data" / "us-cities.csv"
        with cities.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            cls.header = next(reader)
            cls.city_data = [transmogrify(row) for row in reader]

        cls.top_ten = cls.city_data[0:10]

    def test_calc_capacity(self):
        NTICKS = 5 * 365  # 5 years

        # Black Rock Desert, NV = 40°47'13"N 119°12'15"W (40.786944, -119.204167)
        _latitude = 40.786944
        _longitude = -119.204167
        scenario = grid(
            M=10,
            N=10,
            node_size_degs=0.1,
            population_fn=lambda row, col: int(np.random.uniform(10_000, 1_000_000)),
            origin_x=_longitude,
            origin_y=_latitude,
        )

        cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
        birthrates = np.broadcast_to(cbr[None, :], (NTICKS, len(scenario)))  # broadcast (nnodes,) to (nticks, nnodes)

        per_node_extrapolation = calc_capacity(birthrates, scenario.population)
        assert per_node_extrapolation.dtype == np.uint32, f"Expected uint32 dtype, got {per_node_extrapolation.dtype}"
        estimate = per_node_extrapolation.sum()

        assert estimate > scenario.population.sum(), f"Estimate {estimate} not greater than population {scenario.population.sum()}"

        # Run scenarios across two dimensions, iterating over CBRs from [5, 10, 20, 25, 30, 40, 50]
        # Use a scenario with nodes with initial population in [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
        cbrs = np.array([5, 10, 20, 25, 30, 40, 50])
        populations = np.array([10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000])
        previous = None
        for cbr in cbrs:
            tmp = np.array([[cbr]], dtype=np.float32)
            birthrates = np.broadcast_to(tmp, (NTICKS, len(populations)))  # broadcast (1,1) to (nticks, nnodes)

            estimate = calc_capacity(birthrates, populations)  # default safety_factor=1.0

            assert np.all(estimate > populations), f"Estimate \n{estimate}\n not greater than population \n{populations}"

            # Check that estimates increase with increasing CBR
            if previous is not None:
                assert np.all(
                    estimate > previous
                ), f"Estimate \n{estimate}\n with CBR {cbr} not greater than previous estimate at lower CBR \n{previous}"
            previous = estimate

            # Now test with safety factor of 2.0
            safer = calc_capacity(birthrates, populations, safety_factor=2.0)
            assert np.all(
                safer > estimate
            ), f"Estimate with safety factor 2.0 \n{safer}\n not greater than estimate with safety factor 1.0 \n{estimate}"

        return

    def test_calc_capacity_clamps_to_uint32_max(self):
        """Given per-node populations whose projected growth exceeds 2**32 - 1,
        when calc_capacity is called,
        then the overflowing entries are clamped to uint32 max and below-cap entries are unchanged.

        Failure indicates calc_capacity either wraps around (silent overflow into a small value)
        or fails to cap, both of which would propagate corrupt capacities into LaserFrame
        allocations.
        """
        uint32_max = np.iinfo(np.uint32).max  # 4_294_967_295

        # Two nodes: the first would project well above uint32_max under any positive growth;
        # the second is small enough that its projection stays comfortably below the cap.
        # NOTE: passing initial_pop as int64 here because uint32_max + small growth overflows
        # int32 -- the cap-and-cast is exactly the behavior under test.
        initial_pop = np.array([uint32_max, 1_000], dtype=np.int64)

        # 3 years of CBR=35 at every step; modest growth that pushes node-0 over the cap.
        nticks = 3 * 365
        nnodes = 2
        birthrates = np.full((nticks, nnodes), 35.0, dtype=np.float32)

        estimates = calc_capacity(birthrates, initial_pop)

        assert estimates.dtype == np.uint32, f"Expected uint32 dtype, got {estimates.dtype}"
        assert estimates[0] == uint32_max, f"Overflowing entry should clamp to {uint32_max}, got {estimates[0]}"
        assert estimates[1] < uint32_max, f"Below-cap entry should not clamp, got {estimates[1]}"
        assert estimates[1] > initial_pop[1], f"Below-cap entry should grow above initial pop, got {estimates[1]}"

        return

    def test_calc_capacity_rejects_negative_initial_pop(self):
        """Given an ``initial_pop`` array containing one or more negative values,
        when ``calc_capacity`` is called,
        then it raises ``AssertionError`` and the message identifies the offending value(s).

        Failure of this test means a caller could silently pass nonsensical (negative) initial
        populations through ``calc_capacity``. The downstream estimate would then propagate the
        negative product, get rounded, cast to ``uint32`` (wrapping into a huge positive number),
        and corrupt the ``LaserFrame`` capacity allocated from it.
        """
        # Same simple two-node birthrate matrix used by the other calc_capacity tests.
        nticks = 365
        nnodes = 2
        birthrates = np.full((nticks, nnodes), 30.0, dtype=np.float32)

        # Single negative value among otherwise-valid populations.
        initial_pop_one_negative = np.array([10_000, -1], dtype=np.int64)
        with pytest.raises(AssertionError, match=r"Initial populations must be >= 0"):
            calc_capacity(birthrates, initial_pop_one_negative)

        # Multiple negative values: assertion should still fire, and the message should
        # mention at least one of them so a developer can find the bad entry quickly.
        initial_pop_multi_negative = np.array([-5, -42], dtype=np.int64)
        with pytest.raises(AssertionError, match=r"Initial populations must be >= 0.*(-5|-42)"):
            calc_capacity(birthrates, initial_pop_multi_negative)

        # Sanity: zero is allowed (a node with no initial population is a legitimate edge case).
        initial_pop_zero_ok = np.array([0, 10_000], dtype=np.int64)
        estimates = calc_capacity(birthrates, initial_pop_zero_ok)
        assert estimates.dtype == np.uint32
        assert estimates[0] == 0, f"Zero initial pop should stay zero, got {estimates[0]}"
        assert estimates[1] > 10_000, f"Positive initial pop should still grow, got {estimates[1]}"

        return


class TestGridUtilityFunction(unittest.TestCase):
    def check_grid_validity(self, gdf, M, N, node_size_degs=0.1, origin_x=0, origin_y=0):
        assert gdf.shape[0] == M * N, f"Expected {M * N} rows, got {gdf.shape[0]}"
        assert all(
            col in gdf.columns for col in ["nodeid", "population", "geometry"]
        ), f"Expected columns 'nodeid', 'population', 'geometry', got {gdf.columns}"

        assert gdf["nodeid"].min() == 0, f"Expected min nodeid 0, got {gdf['nodeid'].min()}"
        assert gdf["nodeid"].max() == M * N - 1, f"Expected max nodeid {M * N - 1}, got {gdf['nodeid'].max()}"

        assert (
            gdf["geometry"].geom_type.nunique() == 1
        ), f"Expected all geometries to have the same type, got {gdf['geometry'].geom_type.unique()}"
        assert (
            gdf["geometry"].geom_type.unique()[0] == "Polygon"
        ), f"Expected all geometries to be Polygons, got {gdf['geometry'].geom_type.unique()}"

        # Check bounding box: lower left should be (origin_x, origin_y), upper right should be (origin_x + N*node_size_degs, origin_y + M*node_size_degs)
        # 1 degree latitude ~ 111 km, longitude varies but for small grids this is a reasonable check
        minx, miny, maxx, maxy = gdf.total_bounds
        expected_minx = origin_x
        expected_miny = origin_y
        expected_maxx = origin_x + N * node_size_degs
        expected_maxy = origin_y + M * node_size_degs
        assert np.isclose(minx, expected_minx, atol=1e-3), f"Expected minx {expected_minx}, got {minx}"
        assert np.isclose(miny, expected_miny, atol=1e-3), f"Expected miny {expected_miny}, got {miny}"
        assert np.isclose(maxx, expected_maxx, atol=1e-3), f"Expected maxx {expected_maxx}, got {maxx}"
        assert np.isclose(maxy, expected_maxy, atol=1e-3), f"Expected maxy {expected_maxy}, got {maxy}"

        return

    def test_grid_default_population(self):
        M = 4
        N = 5
        node_size_degs = 0.1
        origin_x = -125.0
        origin_y = 25.0

        gdf = grid(M=M, N=N, node_size_degs=node_size_degs, origin_x=origin_x, origin_y=origin_y)

        self.check_grid_validity(gdf, M, N, node_size_degs=node_size_degs, origin_x=origin_x, origin_y=origin_y)
        assert gdf["population"].min() >= 1_000, f"Expected min population >= 1,000, got {gdf['population'].min()}"
        assert gdf["population"].max() <= 100_000, f"Expected max population <= 100,000, got {gdf['population'].max()}"

        return

    def test_horizontal_row(self):
        M = 1
        N = 10
        node_size_degs = 0.1
        origin_x = -125.0
        origin_y = 25.0

        gdf = grid(M=M, N=N, node_size_degs=node_size_degs, origin_x=origin_x, origin_y=origin_y)

        self.check_grid_validity(gdf, M, N, node_size_degs=node_size_degs, origin_x=origin_x, origin_y=origin_y)
        assert gdf["population"].min() >= 1_000, f"Expected min population >= 1,000, got {gdf['population'].min()}"
        assert gdf["population"].max() <= 100_000, f"Expected max population <= 100,000, got {gdf['population'].max()}"

        return

    def test_vertical_column(self):
        M = 10
        N = 1
        node_size_degs = 0.1
        origin_x = -125.0
        origin_y = 25.0

        gdf = grid(M=M, N=N, node_size_degs=node_size_degs, origin_x=origin_x, origin_y=origin_y)
        self.check_grid_validity(gdf, M, N, node_size_degs=node_size_degs, origin_x=origin_x, origin_y=origin_y)
        assert gdf["population"].min() >= 1_000, f"Expected min population >= 1,000, got {gdf['population'].min()}"
        assert gdf["population"].max() <= 100_000, f"Expected max population <= 100,000, got {gdf['population'].max()}"

        return

    def test_grid_custom_population(self):
        M = 4
        N = 5
        node_size_degs = 0.1
        origin_x = -125.0
        origin_y = 25.0

        def custom_population(row: int, col: int) -> int:
            return (row + 1) * (col + 1) * 100

        gdf = grid(M=M, N=N, node_size_degs=node_size_degs, population_fn=custom_population, origin_x=origin_x, origin_y=origin_y)

        self.check_grid_validity(gdf, M, N, node_size_degs=node_size_degs, origin_x=origin_x, origin_y=origin_y)
        assert gdf["population"].min() == 100, f"Expected min population == 100, got {gdf['population'].min()}"
        # max row is M-1, max col is N-1, but the custom population function adds 1 so max population is M*N*100
        assert gdf["population"].max() == (M * N * 100), f"Expected max population == {(M * N * 100)}, got {gdf['population'].max()}"

        return

    def test_grid_invalid_parameters(self):
        with pytest.raises(ValueError, match="M must be >= 1"):
            grid(M=0, N=5)

        with pytest.raises(ValueError, match="N must be >= 1"):
            grid(M=4, N=0)

        with pytest.raises(ValueError, match="node_size_degs must be > 0"):
            grid(M=4, N=5, node_size_degs=0)

        with pytest.raises(ValueError, match=r"node_size_degs must be <= 1.0"):
            grid(M=4, N=5, node_size_degs=2)

        with pytest.raises(ValueError, match="origin_x must be -180 <= origin_x < 180"):
            grid(M=4, N=5, origin_x=-200)

        with pytest.raises(ValueError, match="origin_x must be -180 <= origin_x < 180"):
            grid(M=4, N=5, origin_x=180)

        with pytest.raises(ValueError, match="origin_y must be -90 <= origin_y < 90"):
            grid(M=4, N=5, origin_y=-100)

        with pytest.raises(ValueError, match="origin_y must be -90 <= origin_y < 90"):
            grid(M=4, N=5, origin_y=90)

        def negative_population(row: int, col: int) -> int:
            return -100

        with pytest.raises(ValueError, match="population_fn returned negative population -100 for row 0, col 0"):
            grid(M=4, N=5, population_fn=negative_population)

        return

    def test_grid_default_states_columns(self):
        # Test that the default parameters to grid() return a GeoDataFrame with "S", "E", "I", and "R" columns
        gdf = grid()
        assert "S" in gdf.columns, f"Expected column 'S' in GeoDataFrame columns {gdf.columns}"
        assert np.all(
            gdf["S"] == gdf.population
        ), f"Expected all values in column 'S' to equal population,\n{gdf.population}got\n{gdf['S']}"
        for state in ["E", "I", "R"]:
            assert state in gdf.columns, f"Expected column '{state}' in GeoDataFrame columns {gdf.columns}"
            assert np.all(gdf[state] == 0), f"Expected all values in column '{state}' to be 0, got {gdf[state]}"
        return

    def test_grid_custom_states_columns(self):
        # Test that grid() with custom states returns a GeoDataFrame with those columns
        custom_states = ["sus", "inc", "inf", "rec", "vax"]
        gdf = grid(states=custom_states)
        assert custom_states[0] in gdf.columns, f"Expected column '{custom_states[0]}' in GeoDataFrame columns {gdf.columns}"
        assert np.all(
            gdf[custom_states[0]] == gdf.population
        ), f"Expected all values in column '{custom_states[0]}' to be equal to population,\n{gdf.population}got\n{gdf[custom_states[0]]}"
        for state in custom_states[1:]:
            assert state in gdf.columns, f"Expected column '{state}' in GeoDataFrame columns {gdf.columns}"
            assert np.all(gdf[state] == 0), f"Expected all values in column '{state}' to be 0, got\n{gdf[state]}"
        # Ensure default states are not present
        for default_state in ["S", "E", "I", "R"]:
            assert default_state not in gdf.columns, f"Did not expect column '{default_state}' in GeoDataFrame columns {gdf.columns}"
        return

    def test_initialize_population_exact_counts(self):
        M, N = 3, 2
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 1000, states=states)
        nnodes = M * N
        # Each node: [S, E, I, R] = [900, 50, 30, 20]
        initial = np.tile([900, 50, 30, 20], (nnodes, 1))
        gdf2 = initialize_population(gdf.copy(), initial, states=states)
        for idx, state in enumerate(states):
            assert np.all(gdf2[state] == initial[:, idx]), f"State {state} not set correctly"
        assert np.all(gdf2[states].sum(axis=1) == gdf2.population), "Sum of states does not equal population"

    def test_initialize_population_fractions(self):
        M, N = 2, 2
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 1000, states=states)
        nnodes = M * N
        # Fractions: [S, E, I, R] = [computed, 0.1, 0.2, 0.3]
        fractions = np.tile([0.0, 0.1, 0.2, 0.3], (nnodes, 1))
        gdf2 = initialize_population(gdf.copy(), fractions, states=states)
        # S is computed as the remainder
        expected_S = 1000 - (np.round(0.1 * 1000) + np.round(0.2 * 1000) + np.round(0.3 * 1000))
        assert np.all(gdf2["S"] == expected_S), f"Expected S={expected_S}, got {gdf2['S']}"
        assert np.all(gdf2["E"] == 100), f"Expected E=100, got {gdf2['E']}"
        assert np.all(gdf2["I"] == 200), f"Expected I=200, got {gdf2['I']}"
        assert np.all(gdf2["R"] == 300), f"Expected R=300, got {gdf2['R']}"
        assert np.all(
            gdf2[states].sum(axis=1) == gdf2.population
        ), f"Sum of states\n{gdf2[states].sum(axis=1)}\ndoes not equal population\n{gdf2.population}"

    def test_initialize_population_integer_out_of_range(self):
        M, N = 2, 1
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 100, states=states)
        # Too many people: [S, E, I, R] = [50, 50, 50, 50] = 200 > 100
        initial = np.tile([50, 50, 50, 50], (M * N, 1))
        with pytest.raises(AssertionError, match="Sum of initial states does not equal population at some nodes"):
            initialize_population(gdf.copy(), initial, states=states)

    def test_initialize_population_fractions_sum_gt_one(self):
        M, N = 1, 2
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 100, states=states)
        # Fractions sum to 1.2: [S, E, I, R] = [0.0, 0.5, 0.5, 0.2]
        fractions = np.tile([0.0, 0.5, 0.5, 0.2], (M * N, 1))
        with pytest.raises(ValueError, match="Initial state proportions sum to more than 1.0 at some nodes"):
            initialize_population(gdf.copy(), fractions, states=states)

    def test_initialize_population_invalid_shape(self):
        gdf = grid(M=2, N=2)
        # Wrong shape: should be (4, 4), but is (2, 4)
        initial = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with pytest.raises(ValueError, match="Initial state array shape"):
            initialize_population(gdf.copy(), initial)

    def test_initialize_population_invalid_type(self):
        gdf = grid(M=1, N=1)
        # Mixed types: not all ints, not all floats in [0,1]
        initial = np.array([[1, 0.5, 0, 0]])
        with pytest.raises(ValueError, match="Initial state proportions sum to more than 1.0 at some nodes"):
            initialize_population(gdf.copy(), initial)

    def test_initialize_population_single_row_integer_broadcast(self):
        # Test that a single row of integer values is broadcast to all nodes
        M, N = 3, 2
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 1000, states=states)
        # Provide a single row: [800, 100, 50, 50]
        initial = np.array([[800, 100, 50, 50]])
        gdf2 = initialize_population(gdf.copy(), initial, states=states)
        for idx, state in enumerate(states):
            assert np.all(gdf2[state] == initial[0, idx]), f"State {state} not broadcast correctly"
        assert np.all(
            gdf2[states].sum(axis=1) == gdf2.population
        ), f"Sum of states\n{gdf2[states].sum(axis=1)}\ndoes not equal population\n{gdf2.population}"

    def test_initialize_population_single_row_fraction_broadcast(self):
        # Test that a single row of fractional values is broadcast to all nodes
        M, N = 4, 1
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 500, states=states)
        # Provide a single row: [0.0, 0.2, 0.3, 0.1]
        fractions = np.array([[0.0, 0.2, 0.3, 0.1]])
        gdf2 = initialize_population(gdf.copy(), fractions, states=states)
        expected_E = np.round(0.2 * 500).astype(int)
        expected_I = np.round(0.3 * 500).astype(int)
        expected_R = np.round(0.1 * 500).astype(int)
        expected_S = 500 - (expected_E + expected_I + expected_R)
        assert np.all(gdf2["E"] == expected_E), f"Expected E={expected_E}\ngot\n{gdf2['E']}"
        assert np.all(gdf2["I"] == expected_I), f"Expected I={expected_I}\ngot\n{gdf2['I']}"
        assert np.all(gdf2["R"] == expected_R), f"Expected R={expected_R}\ngot\n{gdf2['R']}"
        assert np.all(gdf2["S"] == expected_S), f"Expected S={expected_S}\ngot\n{gdf2['S']}"
        assert np.all(
            gdf2[states].sum(axis=1) == gdf2.population
        ), f"Sum of states\n{gdf2[states].sum(axis=1)}\ndoes not equal population\n{gdf2.population}"

    def test_initialize_population_exact_counts_from_list(self):
        # Test initialize_population with integer counts provided as a Python list
        M, N = 2, 3
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 500, states=states)
        nnodes = M * N
        # Each node: [S, E, I, R] = [400, 50, 30, 20]
        initial = [[400, 50, 30, 20] for _ in range(nnodes)]
        gdf2 = initialize_population(gdf.copy(), initial, states=states)
        for idx, state in enumerate(states):
            assert np.all(gdf2[state] == initial[0][idx]), f"State {state} not set correctly from list"
        assert np.all(gdf2[states].sum(axis=1) == gdf2.population), "Sum of states does not equal population"

    def test_initialize_population_fractions_from_list(self):
        # Test initialize_population with fractional values provided as a Python list
        M, N = 2, 2
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 1000, states=states)
        nnodes = M * N
        # Fractions: [S, E, I, R] = [computed, 0.15, 0.25, 0.35]
        fractions = [[0.0, 0.15, 0.25, 0.35] for _ in range(nnodes)]
        gdf2 = initialize_population(gdf.copy(), fractions, states=states)
        expected_E = np.round(0.15 * 1000).astype(int)
        expected_I = np.round(0.25 * 1000).astype(int)
        expected_R = np.round(0.35 * 1000).astype(int)
        expected_S = 1000 - (expected_E + expected_I + expected_R)
        assert np.all(gdf2["E"] == expected_E), f"Expected E={expected_E}, got {gdf2['E']}"
        assert np.all(gdf2["I"] == expected_I), f"Expected I={expected_I}, got {gdf2['I']}"
        assert np.all(gdf2["R"] == expected_R), f"Expected R={expected_R}, got {gdf2['R']}"
        assert np.all(gdf2["S"] == expected_S), f"Expected S={expected_S}, got {gdf2['S']}"
        assert np.all(
            gdf2[states].sum(axis=1) == gdf2.population
        ), f"Sum of states\n{gdf2[states].sum(axis=1)}\ndoes not equal population\n{gdf2.population}"

    def test_initialize_population_single_node_exact_counts(self):
        # Test initialize_population with a single node's integer counts (should broadcast)
        M, N = 3, 3
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 200, states=states)
        # Provide a single node's values: [150, 30, 10, 10]
        initial = [150, 30, 10, 10]
        gdf2 = initialize_population(gdf.copy(), initial, states=states)
        for idx, state in enumerate(states):
            assert np.all(gdf2[state] == initial[idx]), f"State {state} not broadcast correctly from single node"
        assert np.all(
            gdf2[states].sum(axis=1) == gdf2.population
        ), f"Sum of states\n{gdf2[states].sum(axis=1)}\ndoes not equal population\n{gdf2.population}"

    def test_initialize_population_single_node_fractions(self):
        # Test initialize_population with a single node's fractions (should broadcast)
        M, N = 2, 4
        states = ["S", "E", "I", "R"]
        gdf = grid(M=M, N=N, node_size_degs=0.1, population_fn=lambda r, c: 400, states=states)
        # Provide a single node's fractions: [0.0, 0.1, 0.2, 0.3]
        fractions = [0.0, 0.1, 0.2, 0.3]
        gdf2 = initialize_population(gdf.copy(), fractions, states=states)
        expected_E = np.round(0.1 * 400).astype(int)
        expected_I = np.round(0.2 * 400).astype(int)
        expected_R = np.round(0.3 * 400).astype(int)
        expected_S = 400 - (expected_E + expected_I + expected_R)
        assert np.all(gdf2["E"] == expected_E), f"Expected E={expected_E}, got {gdf2['E']}"
        assert np.all(gdf2["I"] == expected_I), f"Expected I={expected_I}, got {gdf2['I']}"
        assert np.all(gdf2["R"] == expected_R), f"Expected R={expected_R}, got {gdf2['R']}"
        assert np.all(gdf2["S"] == expected_S), f"Expected S={expected_S}, got {gdf2['S']}"
        assert np.all(
            gdf2[states].sum(axis=1) == gdf2.population
        ), f"Sum of states\n{gdf2[states].sum(axis=1)}\ndoes not equal population\n{gdf2.population}"


if __name__ == "__main__":
    unittest.main()

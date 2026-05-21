import csv
import re
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from laser.core.migration import competing_destinations
from laser.core.migration import distance
from laser.core.migration import gravity
from laser.core.migration import radiation
from laser.core.migration import row_normalizer
from laser.core.migration import stouffer

City = namedtuple("City", ["name", "pop", "lat", "long"])


class TestMigrationFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

        top_ten = cls.city_data[0:10]
        cls.pops = np.array([city.pop for city in top_ten])
        cls.distances = np.array([[distance(city1.lat, city1.long, city2.lat, city2.long) for city2 in top_ten] for city1 in top_ten])

        cls.new_york = cls.city_data[0]
        cls.los_angeles = cls.city_data[1]

        cls.kampala = City(name="Kampala", pop=0, lat=0.3152, long=32.5816)
        cls.nairobi = City(name="Nairobi", pop=0, lat=-1.3032, long=36.8474)

        return

    def test_setup(self):
        """Test the setup of the test class."""
        assert self.header == [
            "City",
            "State",
            "2023 Estimate",
            "2020 Census",
            "Change",
            "mi^2",
            "km^2",
            "density/mi^2",
            "density/km^2",
            "latitude",
            "longitude",
        ], f"us-cities.csv header: {self.header} doesn't match expected header"
        assert len(self.city_data) == 336, f"us-cities.csv has {len(self.city_data)} rows, expected 336"
        assert self.city_data[0] == City(name="New York", pop=8_258_035, lat=40.66, long=-73.94), f"{self.city_data[0].name=} != 'New York'"
        assert self.city_data[1] == City(
            name="Los Angeles", pop=3_820_914, lat=34.02, long=-118.41
        ), f"{self.city_data[1].name=} != 'Los Angeles'"
        assert self.city_data[-1] == City(name="Davenport", pop=100354, lat=41.56, long=-90.6), f"{self.city_data[-1].name=} != 'Davenport'"

        assert len(self.pops) == 10, f"self.pops has {len(self.pops)} elements, expected 10"
        assert self.distances.shape == (10, 10), f"self.distances has shape {self.distances.shape}, expected (10, 10)"
        assert np.all(self.distances.diagonal() == 0), "self.distances.diagonal() != 0"
        assert np.all(self.distances == self.distances.T), "self.distances is not symmetric"

        assert np.isclose(
            self.distances[0, 1], 3957.13675, atol=1e-05
        ), f"New York to Los Angeles distance is {self.distances[0, 1]}, expected 3957.13675"
        assert np.isclose(
            self.distances[0, 9], 1342.25107, atol=1e-05
        ), f"New York to Jacksonville distance is {self.distances[0, 9]}, expected 1342.25107"

        return

    def test_gravity_model(self):
        """Test the gravity model migration function without maximum."""
        (k, a, b, c) = (0.1, 0.5, 1.0, 2.0)
        network = gravity(self.pops, self.distances, k=k, a=a, b=b, c=c)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    dist = k * (self.pops[i] ** a) * (self.pops[j] ** b) * self.distances[i, j] ** (-1 * c)
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

        return

    def test_row_normalizer(self):
        """Test that the row normalizer function works"""

        network = np.array([[0.0, 0.01, 0.02, 0.03], [0.05, 0.0, 0.05, 0.1], [0.2, 0.2, 0.0, 0.6], [0.001, 0.002, 0.003, 0.0]])
        max_rowsum = 0.1
        expected = np.array([[0.0, 0.01, 0.02, 0.03], [0.025, 0.0, 0.025, 0.05], [0.02, 0.02, 0.0, 0.06], [0.001, 0.002, 0.003, 0.0]])
        normalized = row_normalizer(network, max_rowsum)
        assert np.all(np.isclose(normalized, expected)), (
            "mismatch(es): \n\t"
            + "\n\t".join(
                str((idx, normalized[idx], expected[idx])) for idx in zip(*np.where(~np.isclose(normalized, expected)), strict=True)
            ),
        )

        return

    def test_row_normalizer_with_integer_values(self):
        """Test that the row normalizer returns useful values when the input network is an integer data type and the max rowsum is fractional."""

        network = np.array([[0.0, 0.01, 0.02, 0.03], [0.05, 0.0, 0.05, 0.1], [0.2, 0.2, 0.0, 0.6], [0.001, 0.002, 0.003, 0.0]])
        network *= 1000
        network = network.astype(np.int32)
        max_rowsum = 0.1
        expected = np.array(
            [[0.0, 1 / 60, 2 / 60, 3 / 60], [0.025, 0.0, 0.025, 0.05], [0.02, 0.02, 0.0, 0.06], [1 / 60, 2 / 60, 3 / 60, 0.0]]
        )
        normalized = row_normalizer(network, max_rowsum)
        assert np.all(np.isclose(normalized, expected)), (
            "mismatch(es): \n\t"
            + "\n\t".join(
                str((idx, normalized[idx], expected[idx])) for idx in zip(*np.where(~np.isclose(normalized, expected)), strict=True)
            ),
        )

        return

    def test_competing_destinations(self):
        """Test the competing destinations migration function."""
        (k, a, b, c, delta) = (1.0, 0.5, 1.0, 2.0, 0.5)
        network = competing_destinations(self.pops, self.distances, k=k, a=a, b=b, c=c, delta=delta)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    competition = 0
                    for m in range(len(self.pops)):
                        if m != i and m != j:
                            competition = competition + ((self.pops[m] ** b) * self.distances[j][m] ** (-1 * c))
                    dist = k * (self.pops[i] ** a) * (self.pops[j] ** b) * (self.distances[i, j] ** (-1 * c)) * ((competition) ** delta)
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

    def test_stouffer_exclude_home(self):
        """Test the Stouffer migration function, excluding home."""
        (k, a, b, include_home) = (0.1, 0.5, 1.0, False)
        network = stouffer(self.pops, self.distances, k=k, a=a, b=b, include_home=include_home)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    mysum = np.int64(0)
                    for m in range(len(self.pops)):
                        if (m != i) and (self.distances[i][m] <= self.distances[i][j]):
                            mysum = mysum + self.pops[m]
                    dist = k * (self.pops[i] ** a) * (self.pops[j] / mysum) ** b
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

    def test_stouffer_include_home(self):
        """Test the Stouffer migration function, including home."""
        (k, a, b, include_home) = (0.1, 0.5, 1.0, True)
        network = stouffer(self.pops, self.distances, k=k, a=a, b=b, include_home=include_home)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    mysum = np.int64(0)
                    for m in range(len(self.pops)):
                        if self.distances[i][m] <= self.distances[i][j]:
                            mysum = mysum + self.pops[m]
                    dist = k * (self.pops[i] ** a) * (self.pops[j] / mysum) ** b
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

    def test_stouffer_equidistant_nodes(self):
        """Test that we have appropriately handled the Stouffer model when there are multiple nodes equidistant from a source."""
        (k, a, b, include_home) = (0.1, 0.5, 1.0, True)
        pops = np.array([100000.0, 20000.0, 60000.0, 200000.0])
        distances = np.array(
            [[0.0, 100.0, 200.0, 300.0], [100.0, 0.0, 200.0, 300.0], [200.0, 200.0, 0.0, 100.0], [300.0, 300.0, 100.0, 0.0]]
        )

        network = stouffer(pops, distances, k=k, a=a, b=b, include_home=include_home)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    mysum = np.int64(0)
                    for m in range(len(pops)):
                        if distances[i][m] <= distances[i][j]:
                            mysum = mysum + pops[m]
                    dist = k * (pops[i] ** a) * (pops[j] / mysum) ** b
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

    def test_radiation_exclude_home(self):
        """Test the radiation migration function, excluding home."""
        (k, include_home) = (0.1, False)
        network = radiation(self.pops, self.distances, k=k, include_home=include_home)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    mysum = np.int64(0)
                    for m in range(len(self.pops)):
                        if (m != i) and (self.distances[i][m] <= self.distances[i][j]):
                            mysum = mysum + self.pops[m]
                    dist = k * self.pops[i] * self.pops[j] / ((self.pops[i] + mysum) * (self.pops[i] + self.pops[j] + mysum))
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

    def test_radiation_include_home(self):
        """Test the radiation migration function, including home."""
        (k, include_home) = (0.1, True)
        network = radiation(self.pops, self.distances, k=k, include_home=include_home)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    mysum = np.int64(0)
                    for m in range(len(self.pops)):
                        if self.distances[i][m] <= self.distances[i][j]:
                            mysum = mysum + self.pops[m]
                    dist = k * self.pops[i] * self.pops[j] / ((self.pops[i] + mysum) * (self.pops[i] + self.pops[j] + mysum))
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

    def test_radiation_equidistant_nodes(self):
        """Test that we have appropriately handled the radiation model when there are multiple nodes equidistant from a source."""
        (k, include_home) = (0.1, False)
        pops = np.array([100000.0, 20000.0, 60000.0, 200000.0])
        distances = np.array(
            [[0.0, 100.0, 200.0, 300.0], [100.0, 0.0, 200.0, 300.0], [200.0, 200.0, 0.0, 100.0], [300.0, 300.0, 100.0, 0.0]]
        )

        network = radiation(pops, distances, k=k, include_home=include_home)
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j:
                    mysum = np.int64(0)
                    for m in range(len(pops)):
                        if (m != i) and (distances[i][m] <= distances[i][j]):
                            mysum = mysum + pops[m]
                    dist = k * pops[i] * pops[j] / ((pops[i] + mysum) * (pops[i] + pops[j] + mysum))
                    assert np.isclose(network[i, j], dist), f"network[{i}, {j}] = {network[i, j]}, expected {dist=}"
                else:
                    assert network[i, j] == 0, f"network[{i}, {j}] = {network[i, j]}, expected 0"

    def test_distance_one_degree_longitude(self):
        """Test the distance function for one degree of longitude."""
        assert np.isclose(
            distance(lat1=0, lon1=0, lat2=0, lon2=1), 111.19493, atol=1e-05
        ), f"1 degree longitude distance is {distance(lat1=0, lon1=0, lat2=0, lon2=1)}, expected 111.19493km"  # 1cm

    def test_distance_one_degree_latitude(self):
        """Test the distance function for one degree of latitude."""
        assert np.isclose(
            distance(lat1=0, lon1=0, lat2=1, lon2=0), 111.19493, atol=1e-05
        ), f"1 degree latitude distance is {distance(lat1=0, lon1=0, lat2=1, lon2=0)}, expected 111.19493km"  # 1cm

    def test_distance_nyc_la(self):
        """Test the distance function for New York City to Los Angeles."""
        assert np.isclose(
            d := distance(lat1=self.new_york.lat, lon1=self.new_york.long, lat2=self.los_angeles.lat, lon2=self.los_angeles.long),
            3957.13675,
            atol=1e-05,
        ), f"NYC to LA distance is {d}, expected 3957.13675km"  # 1cm

    def test_distance_across_equator(self):
        """Test the distance function for crossing the equator."""
        assert np.isclose(
            d := distance(lat1=self.kampala.lat, lon1=self.kampala.long, lat2=self.nairobi.lat, lon2=self.nairobi.long),
            507.29393,
            atol=1e-05,
        ), f"Kampala to Nairobi (crossing the equator) distance is {d}, expected 507.29393km"  # 1cm

        return

    def test_distance_with_arrays(self):
        """Test the distance function with arrays."""
        lat1 = np.array([0, self.new_york.lat, self.kampala.lat])  # 0, New York, Kampala
        lon1 = np.array([0, self.new_york.long, self.kampala.long])  # 0, New York, Kampala
        lat2 = np.array([0, 1, self.los_angeles.lat, self.nairobi.lat])  # 0, 1, Los Angeles, Nairobi
        lon2 = np.array([1, 0, self.los_angeles.long, self.nairobi.long])  # 1, 0, Los Angeles, Nairobi
        distances = distance(lat1, lon1, lat2, lon2)
        expected = np.array(
            [
                [111.19493, 111.19493, 12590.05805, 4099.44246],  # (0,0)    to [(0,1), (1,0), Los Angeles, Nairobi]
                [8743.51340, 8586.53448, 3957.13675, 11841.98183],  # New York to [(0,1), (1,0), Los Angeles, Nairobi]  NY-Nairobi 11,844km
                [
                    3511.87051,
                    3623.44186,
                    15144.87664,
                    507.29393,
                ],  # Kampala  to [(0,1), (1,0), Los Angeles, Nairobi]  Kampala-Los Angeles 15,136km
            ]
        )
        assert np.allclose(
            distances, expected, atol=1e-05
        ), f"distance({lat1=}, {lon1=}, {lat2=}, {lon2=}) = {distances}, expected {expected=}"

        return

    def test_distance_scalar_array(self):
        top_ten = self.city_data[0:10]
        (lat1, lon1) = top_ten[0].lat, top_ten[0].long  # New York
        lat2 = np.array([city.lat for city in top_ten])
        lon2 = np.array([city.long for city in top_ten])
        distances = distance(lat1, lon1, lat2, lon2)
        expected = np.array(
            [0.00000, 3957.13675, 1154.88510, 2283.37528, 3444.88038, 124.08572, 2547.74127, 3906.51957, 2206.50912, 1342.25107]
        )
        assert np.allclose(
            distances, expected, atol=1e-05
        ), f"distance({lat1=}, {lon1=}, {lat2=}, {lon2=}) = {distances}, expected {expected=}"

        return

    # TODO? - test with array lat1/lon1 and scalar lat2/lon2
    def test_distance_array_scalar(self):
        top_ten = self.city_data[0:10]
        lat1 = np.array([city.lat for city in top_ten])
        lon1 = np.array([city.long for city in top_ten])
        (lat2, lon2) = top_ten[0].lat, top_ten[0].long  # New York
        distances = distance(lat1, lon1, lat2, lon2)
        expected = np.array(
            [0.00000, 3957.13675, 1154.88510, 2283.37528, 3444.88038, 124.08572, 2547.74127, 3906.51957, 2206.50912, 1342.25107]
        )
        assert np.allclose(
            distances, expected, atol=1e-05
        ), f"distance({lat1=}, {lon1=}, {lat2=}, {lon2=}) = {distances}, expected {expected=}"

        return

    def test_distance_with_self(self):
        """Test the distance function with only lat1 and lon1 provided."""
        top_ten = self.city_data[0:10]
        lat1 = np.array([city.lat for city in top_ten])
        lon1 = np.array([city.long for city in top_ten])
        distances = distance(lat1, lon1)
        expected = self.distances
        assert np.allclose(distances, expected, atol=1e-05), f"distance({lat1=}, {lon1=}) = {distances}, expected {expected=}"

        return

    def test_distance_invalid_arguments(self):
        """Test the distance function with invalid arguments."""
        lat1 = "0"
        with pytest.raises(TypeError, match=re.escape(f"lat1 must be a numeric value or NumPy array ({type(lat1)=})")):
            distance(lat1, 0, 0, 1)

        lon1 = "0"
        with pytest.raises(TypeError, match=re.escape(f"lon1 must be a numeric value or NumPy array ({type(lon1)=})")):
            distance(0, lon1, 0, 1)

        lat2 = "0"
        with pytest.raises(TypeError, match=re.escape(f"lat2 must be a numeric value or NumPy array ({type(lat2)=})")):
            distance(0, 0, lat2, 1)

        lon2 = "1"
        with pytest.raises(TypeError, match=re.escape(f"lon2 must be a numeric value or NumPy array ({type(lon2)=})")):
            distance(0, 0, 0, lon2)

        # Additional shape mismatch tests for distance function

        # lat1 and lon1 shape mismatch
        lat1 = np.array([0, 1])
        lon1 = np.array([0, 1, 2])
        lat2 = np.array([0, 1])
        lon2 = np.array([0, 1])
        with pytest.raises(TypeError, match=r"lat1 and lon1 must have the same shape"):
            distance(lat1, lon1, lat2, lon2)

        # lat2 and lon2 shape mismatch
        lat1 = np.array([0, 1])
        lon1 = np.array([0, 1])
        lat2 = np.array([0, 1, 2])
        lon2 = np.array([0, 1])
        with pytest.raises(TypeError, match=r"lat2 and lon2 must have the same shape"):
            distance(lat1, lon1, lat2, lon2)

        top_ten = self.city_data[0:10]
        lat1 = np.array([city.lat for city in top_ten])
        lon1 = np.array([city.long for city in top_ten])
        with pytest.raises(ValueError, match=r"Either both or neither of lat2 and lon2 must be provided."):
            distance(lat1, lon1, None, lon1)

        with pytest.raises(ValueError, match=r"Either both or neither of lat2 and lon2 must be provided."):
            distance(lat1, lon1, lat1, None)

        return


class TestMigrationFunctionSanityChecks(unittest.TestCase):
    def test_gravity_model_sanity(self):
        """Test the gravity model migration function with maximum."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            gravity(pops, np.ones((3, 3)), 1, 1, 1, 1)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            gravity(pops, np.ones((3, 3)), 1, 1, 1, 1)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            gravity(pops, np.ones((3, 3)), 1, 1, 1, 1)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            gravity(np.array([-1, 2, 3]), np.ones((3, 3)), 1, 1, 1, 1)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"distances must be a square matrix with length equal to the length of pops ({distances.shape=}, {pops.shape=})"
            ),
        ):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1)

        # Test that distance parameter is a non-negative array
        with pytest.raises(ValueError, match="distances must contain only non-negative values"):
            gravity(np.array([1, 2, 3]), np.array([[-1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 1, 1, 1)

        # Test that distance parameter is a symmetric array
        with pytest.raises(ValueError, match="distances must be a symmetric matrix"):
            gravity(np.array([1, 2, 3]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 1, 1, 1)

        # Test that k parameter is a numeric value
        k = "1"
        with pytest.raises(TypeError, match=re.escape(f"k must be a numeric value ({type(k)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, 1)

        # Test that k parameter is a non-negative value
        k = -1
        with pytest.raises(ValueError, match=re.escape(f"k must be a non-negative value ({k=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, 1)

        # Test that a parameter is a numeric value
        a = "1"
        with pytest.raises(TypeError, match=re.escape(f"a must be a numeric value ({type(a)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, 1)

        # Test that a is a non-negative value
        a = -1
        with pytest.raises(ValueError, match=re.escape(f"a must be a non-negative value ({a=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, 1)

        # Test that b parameter is a numeric value
        b = "1"
        with pytest.raises(TypeError, match=re.escape(f"b must be a numeric value ({type(b)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, 1)

        # Test that b is a non-negative value
        b = -1
        with pytest.raises(ValueError, match=re.escape(f"b must be a non-negative value ({b=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, 1)

        # Test that c parameter is a numeric value
        c = "1"
        with pytest.raises(TypeError, match=re.escape(f"c must be a numeric value ({type(c)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, c)

        # Test that c is a non-negative value
        c = -1
        with pytest.raises(ValueError, match=re.escape(f"c must be a non-negative value ({c=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, c)

        return

    def test_row_normalizer_sanity(self):
        """Test the row normalization function."""

        # Test that network parameter is a NumPy array
        network = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"network must be a NumPy array ({type(network)=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a numeric array
        network = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"network must be a numeric array ({network.dtype=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a 2D array
        network = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"network must be a 2D array ({network.shape=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a square array
        network = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(TypeError, match=re.escape(f"network must be a square matrix ({network.shape=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a non-negative array
        with pytest.raises(ValueError, match="network must contain only non-negative values"):
            row_normalizer(np.array([[-1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0.1)

        # Test that max_rowsum parameter is a numeric value
        max_rowsum = "0.1"
        with pytest.raises(TypeError, match=re.escape(f"max_rowsum must be a numeric value ({type(max_rowsum)=})")):
            row_normalizer(np.full((3, 3), 0.0625), max_rowsum)

        # Test that max_rowsum is a non-negative value
        with pytest.raises(ValueError, match=re.escape("max_rowsum must be in [0, 1]")):
            row_normalizer(np.full((3, 3), 0.0625), -0.1)

        # Test that max_rowsum is less than or equal to 1
        with pytest.raises(ValueError, match=re.escape("max_rowsum must be in [0, 1]")):
            row_normalizer(np.full((3, 3), 0.0625), 1.1)

        return

    def test_competing_destinations_sanity(self):
        """Test the competing destinations migration function."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            competing_destinations(pops, np.ones((3, 3)), 1, 1, 1, 1, 1)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            competing_destinations(pops, np.ones((3, 3)), 1, 1, 1, 1, 1)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            competing_destinations(pops, np.ones((3, 3)), 1, 1, 1, 1, 1)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            competing_destinations(np.array([-1, 2, 3]), np.ones((3, 3)), 1, 1, 1, 1, 1)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1, 1, 1)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1, 1, 1)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"distances must be a square matrix with length equal to the length of pops ({distances.shape=}, {pops.shape=})"
            ),
        ):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1, 1, 1)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1, 1, 1)

        # Test that distance parameter is a non-negative array
        with pytest.raises(ValueError, match="distances must contain only non-negative values"):
            competing_destinations(np.array([1, 2, 3]), np.array([[-1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 1, 1, 1, 1)

        # Test that k parameter is a numeric value
        k = "1"
        with pytest.raises(TypeError, match=re.escape(f"k must be a numeric value ({type(k)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, 1, 1)

        # Test that k parameter is a non-negative value
        k = -1
        with pytest.raises(ValueError, match=re.escape(f"k must be a non-negative value ({k=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, 1, 1)

        # Test that a parameter is a numeric value
        a = "1"
        with pytest.raises(TypeError, match=re.escape(f"a must be a numeric value ({type(a)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, 1, 1)

        # Test that a is a non-negative value
        a = -1
        with pytest.raises(ValueError, match=re.escape(f"a must be a non-negative value ({a=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, 1, 1)

        # Test that parameter b is a numeric value
        b = "1"
        with pytest.raises(TypeError, match=re.escape(f"b must be a numeric value ({type(b)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, 1, 1)

        # Test that b is a non-negative value
        b = -1
        with pytest.raises(ValueError, match=re.escape(f"b must be a non-negative value ({b=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, 1, 1)

        # Test that parameter c is a numeric value
        c = "1"
        with pytest.raises(TypeError, match=re.escape(f"c must be a numeric value ({type(c)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, c, 1)

        # Test that c is a non-negative value
        c = -1
        with pytest.raises(ValueError, match=re.escape(f"c must be a non-negative value ({c=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, c, 1)

        # Test that parameter delta is a numeric value
        delta = "1"
        with pytest.raises(TypeError, match=re.escape(f"delta must be a numeric value ({type(delta)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, 1, delta)

        return

    def test_stouffer_sanity(self):
        """Test the Stouffer migration function."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            stouffer(pops, np.ones((3, 3)), 1, 1, 1, False)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            stouffer(pops, np.ones((3, 3)), 1, 1, 1, False)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            stouffer(pops, np.ones((3, 3)), 1, 1, 1, False)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            stouffer(np.array([-1, 2, 3]), np.ones((3, 3)), 1, 1, 1, False)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"distances must be a square matrix with length equal to the length of pops ({distances.shape=}, {pops.shape=})"
            ),
        ):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that parameter k is a numeric value
        k = "1"
        with pytest.raises(TypeError, match=re.escape(f"k must be a numeric value ({type(k)=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, False)

        # Test that k is a non-negative value
        k = -1
        with pytest.raises(ValueError, match=re.escape(f"k must be a non-negative value ({k=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, False)

        # Test that parameter a is a numeric value
        a = "1"
        with pytest.raises(TypeError, match=re.escape(f"a must be a numeric value ({type(a)=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, False)

        # Test that a is a non-negative value
        a = -1
        with pytest.raises(ValueError, match=re.escape(f"a must be a non-negative value ({a=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, False)

        # Test that parameter b is a numeric value
        b = "1"
        with pytest.raises(TypeError, match=re.escape(f"b must be a numeric value ({type(b)=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, False)

        # Test that b is a non-negative value
        b = -1
        with pytest.raises(ValueError, match=re.escape(f"b must be a non-negative value ({b=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, False)

        return

    def test_radiation_sanity(self):
        """Test the radiation migration function."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            radiation(pops, np.ones((3, 3)), 1, False)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            radiation(pops, np.ones((3, 3)), 1, False)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            radiation(pops, np.ones((3, 3)), 1, False)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            radiation(np.array([-1, 2, 3]), np.ones((3, 3)), 1, False)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"distances must be a square matrix with length equal to the length of pops ({distances.shape=}, {pops.shape=})"
            ),
        ):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that parameter k is a numeric value
        k = "1"
        with pytest.raises(TypeError, match=re.escape(f"k must be a numeric value ({type(k)=})")):
            radiation(np.array([1, 2, 3]), np.ones((3, 3)), k, False)

        # Test that k is a non-negative value
        k = -1
        with pytest.raises(ValueError, match=re.escape(f"k must be a non-negative value ({k=})")):
            radiation(np.array([1, 2, 3]), np.ones((3, 3)), k, False)

        return

    def test_distance_sanity(self):
        """Test the distance function."""

        # Test that lat1 parameter is a numeric value
        with pytest.raises(TypeError, match="lat1 must be a numeric value or NumPy array"):
            distance("0", 0, 0, 0)

        # Test that lat1 is a valid latitude
        with pytest.raises(ValueError, match=re.escape("lat1 must be in the range [-90, 90]")):
            distance(91, 0, 0, 0)

        # Test that lon1 parameter is a numeric value
        with pytest.raises(TypeError, match="lon1 must be a numeric value or NumPy array"):
            distance(0, "0", 0, 0)

        # Test that lon1 is a valid longitude
        with pytest.raises(ValueError, match=re.escape("lon1 must be in the range [-180, 180]")):
            distance(0, 181, 0, 0)

        # Test that lat2 parameter is a numeric value
        with pytest.raises(TypeError, match="lat2 must be a numeric value or NumPy array"):
            distance(0, 0, "0", 0)

        # Test that lat2 is a valid latitude
        with pytest.raises(ValueError, match=re.escape("lat2 must be in the range [-90, 90]")):
            distance(0, 0, 91, 0)

        # Test that lon2 parameter is a numeric value
        with pytest.raises(TypeError, match="lon2 must be a numeric value or NumPy array"):
            distance(0, 0, 0, "0")

        # Test that lon2 is a valid longitude
        with pytest.raises(ValueError, match=re.escape("lon2 must be in the range [-180, 180]")):
            distance(0, 0, 0, 181)

        return


class TestsForOverflow(unittest.TestCase):
    pops = np.array([99510, 595855, 263884], dtype=np.int32)
    dist = np.array([[0, 4, 66], [4, 0, 827], [66, 827, 0]], dtype=np.int32)
    k = 3000
    a = 1
    b = 1
    c = 2.0

    def test_gravity(self):
        g = gravity(self.pops, self.dist, self.k, self.a, self.b, self.c)
        assert np.all(g >= 0), "Gravity calculation should not overflow and/or return negative values."

        return

    def test_competing_destinations(self):
        cd = competing_destinations(self.pops, self.dist, self.k, self.a, self.b, self.c, delta=1.1)
        assert np.all(cd >= 0), "Competing destinations calculation should not overflow and/or return negative values."

        return

    def test_stouffer(self):
        s = stouffer(self.pops, self.dist, self.k, self.a, self.b, include_home=False)
        assert np.all(s >= 0), "Stouffer migration calculation (include_home=False) should not overflow and/or return negative values."
        s = stouffer(self.pops, self.dist, self.k, self.a, self.b, include_home=True)
        assert np.all(s >= 0), "Stouffer migration calculation (include_home=True) should not overflow and/or return negative values."

        return

    def test_radiation(self):
        r = radiation(self.pops, self.dist, self.k, include_home=False)
        assert np.all(r >= 0), "Radiation migration calculation (include_home=False) should not overflow and/or return negative values."
        r = radiation(self.pops, self.dist, self.k, include_home=True)
        assert np.all(r >= 0), "Radiation migration calculation (include_home=True) should not overflow and/or return negative values."

        return


class TestMigrationVectorizationRegression(unittest.TestCase):
    """Regression tests pinning the vectorized implementations of competing_destinations and distance.

    Given an N-node spatial input, when the vectorized helpers run, then their output must
    match an independent reference implemented in this test file using plain Python loops.

    A failure here implies the vectorization in `migration.py` diverges from the original
    semantics, which would silently break downstream calibrations and published-result
    reproducibility.
    """

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(seed=20260514)
        n = 8
        cls.pops = rng.integers(10_000, 1_000_000, size=n).astype(np.float64)
        cls.lat = rng.uniform(-60.0, 60.0, size=n)
        cls.lon = rng.uniform(-150.0, 150.0, size=n)
        cls.distances = distance(cls.lat, cls.lon)
        return

    @staticmethod
    def _reference_distance(lat1, lon1, lat2, lon2):
        """Loop-based reference Haversine; mirrors the prior implementation exactly."""
        lat1 = np.radians(np.asarray(lat1).flatten())
        lon1 = np.radians(np.asarray(lon1).flatten())
        lat2 = np.radians(np.asarray(lat2).flatten())
        lon2 = np.radians(np.asarray(lon2).flatten())
        d = np.zeros((lat1.size, lat2.size))
        for i in range(lat1.size):
            dlat = lat2 - lat1[i]
            dlon = lon2 - lon1[i]
            d[i, :] = np.sin(dlat / 2) ** 2 + np.cos(lat1[i]) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(d))

    @staticmethod
    def _reference_competing_destinations(pops, distances, k, a, b, c, delta):
        """Loop-based reference for competing_destinations; mirrors the prior implementation exactly."""
        pops = pops.astype(np.float64)
        n = len(pops)
        net = gravity(pops, distances, k=k, a=a, b=b, c=c)

        distances1 = distances.copy().astype(np.float64)
        np.fill_diagonal(distances1, 1)
        comp = pops**b * distances1 ** (-1 * c)
        np.fill_diagonal(comp, 0)
        mysums = np.sum(comp, axis=1)

        for i in range(n):
            for j in range(n):
                if j != i:
                    net[i][j] = net[i][j] * (mysums[j] - comp[j][i]) ** delta
        np.fill_diagonal(net, 0)
        return net

    def test_distance_matches_loop_reference(self):
        """Given lat/lon vectors, when vectorized distance runs, then it matches the looped reference within float tolerance."""
        actual = distance(self.lat, self.lon)
        expected = self._reference_distance(self.lat, self.lon, self.lat, self.lon)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)

    def test_distance_rectangular_matches_loop_reference(self):
        """Given different-sized lat/lon vectors, when vectorized distance runs, then NxM output matches the looped reference."""
        lat2 = self.lat[:4]
        lon2 = self.lon[:4]
        actual = distance(self.lat, self.lon, lat2, lon2)
        expected = self._reference_distance(self.lat, self.lon, lat2, lon2)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)

    def test_competing_destinations_matches_loop_reference(self):
        """Given a multi-node scenario, when vectorized competing_destinations runs, then output equals the looped reference within float tolerance."""
        k, a, b, c, delta = 0.01, 1.0, 1.0, 2.0, -0.1
        actual = competing_destinations(self.pops, self.distances, k=k, a=a, b=b, c=c, delta=delta)
        expected = self._reference_competing_destinations(self.pops, self.distances, k, a, b, c, delta)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

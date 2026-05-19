import unittest

import numba as nb
import numpy as np

import laser.core.random as random
from laser.core.random import get_seed
from laser.core.random import prng as rng
from laser.core.random import seed


# Make some calls to various random number generators with Numba threads.
@nb.njit((nb.uint64, nb.int64[:], nb.float64[:], nb.float64[:], nb.uint64[:]), nogil=True, parallel=True)
def noise(count, integers, floats, normal, poisson):  # pragma: no cover
    for i in nb.prange(count):
        integers[i] = np.random.randint(0, 100)
        floats[i] = np.random.random()
        normal[i] = np.random.normal()
        poisson[i] = np.random.poisson(5)

    return


# Make some calls to the Numba-fied random number generator in non-parallel mode.
@nb.njit((nb.uint64, nb.int64[:]), nogil=True)
def nbintegers(count, integers):  # pragma: no cover
    for i in range(count):
        integers[i] = np.random.randint(0, 100)

    return


class TestRandomSeed(unittest.TestCase):
    def test_numpy_seed(self):
        """Test that setting the random seed applies to the built-in NumPy generator."""
        np.random.seed(20241009)
        integers1 = np.random.randint(0, 100, 10)
        floats1 = np.random.random(10)
        normal1 = np.random.normal(0, 1, 10)
        poisson1 = np.random.poisson(5, 10)
        np.random.seed(20241009)
        integers2 = np.random.randint(0, 100, 10)
        floats2 = np.random.random(10)
        normal2 = np.random.normal(0, 1, 10)
        poisson2 = np.random.poisson(5, 10)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)

        return

    def test_prng_seed(self):
        """Test that setting the random seed applies to the laser-core "global" prng."""
        prng = seed(20241009)
        integers1 = prng.integers(0, 100, 10)
        floats1 = prng.random(10)
        normal1 = prng.normal(0, 1, 10)
        poisson1 = prng.poisson(5, 10)
        prng = seed(20241009)
        integers2 = prng.integers(0, 100, 10)
        floats2 = prng.random(10)
        normal2 = prng.normal(0, 1, 10)
        poisson2 = prng.poisson(5, 10)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)

        return

    def test_rng_seed(self):
        """Test that setting the random seed applies equally to the prng returned from `seed()` and from `rng()`."""
        prng = seed(20241009)
        integers1 = prng.integers(0, 100, 10)
        floats1 = prng.random(10)
        normal1 = prng.normal(0, 1, 10)
        poisson1 = prng.poisson(5, 10)
        _ = seed(20241009)
        prng = rng()
        integers2 = prng.integers(0, 100, 10)
        floats2 = prng.random(10)
        normal2 = prng.normal(0, 1, 10)
        poisson2 = prng.poisson(5, 10)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)

        return

    def test_numba_seed(self):
        """Test that setting the random seed applies to Numba-fied functions."""
        count = 1024
        seed(20241009)
        integers1 = np.empty(count, dtype=np.int64)
        floats1 = np.empty(count, dtype=np.float64)
        normal1 = np.empty(count, dtype=np.float64)
        poisson1 = np.empty(count, dtype=np.uint64)
        noise(np.uint64(count), integers1, floats1, normal1, poisson1)
        integers1a = np.array(integers1)
        nbintegers(1024, integers1a)
        seed(20241009)
        integers2 = np.empty(count, dtype=np.int64)
        floats2 = np.empty(count, dtype=np.float64)
        normal2 = np.empty(count, dtype=np.float64)
        poisson2 = np.empty(count, dtype=np.uint64)
        noise(np.uint64(count), integers2, floats2, normal2, poisson2)
        integers2a = np.array(integers2)
        nbintegers(1024, integers2a)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)
        assert np.array_equal(integers1a, integers2a)

        return

    def test_get_seed(self):
        """Test that the seed returned by `get_seed()` matches the seed set by `seed()`."""
        seed(20241009)
        assert get_seed() == 20241009

        return

    def test_uninitialized_prng(self):
        random._seed = None
        random._prng = None
        prng = rng()
        assert prng is not None, "Calling laser.core.random.prng() should initialize the LASER global prng."
        assert get_seed() is not None, "Calling laser.core.random.prng() should initialize the LASER global prng seed."

        return

    def test_prng_mix(self):
        # Call a mix of NumPy default random functions, prng functions, and Numba-fied functions
        # with a mix of random draw counts.
        # Verify that re-seeding results in the same random draws.
        which = np.random.randint(0, 16, 1024)
        counts = np.random.randint(1, 1025, 1024)
        prng = seed(20241015)
        functions = [
            lambda count: np.random.randint(0, 128, count),
            lambda count: np.random.random(count),
            lambda count: np.random.normal(0, 1, count),
            lambda count: np.random.poisson(5, count),
            lambda count: prng.integers(0, 128, count),
            lambda count: prng.random(count),
            lambda count: prng.normal(0, 1, count),
            lambda count: prng.poisson(5, count),
            lambda count: _nbintegers(count),
            lambda count: _nbfloats(count),
            lambda count: _nbnormal(count),
            lambda count: _nbpoisson(count),
            lambda count: _nbpbintegers(count),
            lambda count: _nbpbfloats(count),
            lambda count: _nbpbnormal(count),
            lambda count: _nbpbpoisson(count),
        ]
        results1 = [functions[which[i]](counts[i]) for i in range(1024)]
        prng = seed(20241015)
        results2 = [functions[which[i]](counts[i]) for i in range(1024)]
        for result1, result2 in zip(results1, results2, strict=True):
            assert np.array_equal(result1, result2)

        return

    def test_prng_mix_uninitialized(self):
        # Call a mix of NumPy default random functions, prng functions, and Numba-fied functions
        # with a mix of random draw counts.
        # Verify that re-seeding results in the same random draws.
        which = np.random.randint(0, 16, 1024)
        counts = np.random.randint(1, 1025, 1024)
        random._prng = None  # Reset the LASER global prng
        prng = rng()
        functions = [
            lambda count: np.random.randint(0, 128, count),
            lambda count: np.random.random(count),
            lambda count: np.random.normal(0, 1, count),
            lambda count: np.random.poisson(5, count),
            lambda count: prng.integers(0, 128, count),
            lambda count: prng.random(count),
            lambda count: prng.normal(0, 1, count),
            lambda count: prng.poisson(5, count),
            lambda count: _nbintegers(count),
            lambda count: _nbfloats(count),
            lambda count: _nbnormal(count),
            lambda count: _nbpoisson(count),
            lambda count: _nbpbintegers(count),
            lambda count: _nbpbfloats(count),
            lambda count: _nbpbnormal(count),
            lambda count: _nbpbpoisson(count),
        ]
        results1 = [functions[which[i]](counts[i]) for i in range(1024)]
        prng = seed(get_seed())
        results2 = [functions[which[i]](counts[i]) for i in range(1024)]
        for result1, result2 in zip(results1, results2, strict=True):
            assert np.array_equal(result1, result2)

        return


@nb.njit((nb.uint64,))
def _nbintegers(count):  # pragma: no cover
    result = np.empty(count, dtype=np.int64)
    for i in range(count):
        result[i] = np.random.randint(0, 100)

    return result


@nb.njit((nb.uint64,))
def _nbfloats(count):  # pragma: no cover
    result = np.empty(count, dtype=np.float64)
    for i in range(count):
        result[i] = np.random.random()

    return result


@nb.njit((nb.uint64,))
def _nbnormal(count):  # pragma: no cover
    result = np.empty(count, dtype=np.float64)
    for i in range(count):
        result[i] = np.random.normal(0, 1)

    return result


@nb.njit((nb.uint64,))
def _nbpoisson(count):  # pragma: no cover
    result = np.empty(count, dtype=np.uint64)
    for i in range(count):
        result[i] = np.random.poisson(5)

    return result


@nb.njit((nb.uint64,), parallel=True)
def _nbpbintegers(count):  # pragma: no cover
    result = np.empty(count, dtype=np.int64)
    for i in nb.prange(count):
        result[i] = np.random.randint(0, 100)

    return result


@nb.njit((nb.uint64,), parallel=True)
def _nbpbfloats(count):  # pragma: no cover
    result = np.empty(count, dtype=np.float64)
    for i in nb.prange(count):
        result[i] = np.random.random()

    return result


@nb.njit((nb.uint64,), parallel=True)
def _nbpbnormal(count):  # pragma: no cover
    result = np.empty(count, dtype=np.float64)
    for i in nb.prange(count):
        result[i] = np.random.normal(0, 1)

    return result


@nb.njit((nb.uint64,), parallel=True)
def _nbpbpoisson(count):  # pragma: no cover
    result = np.empty(count, dtype=np.uint64)
    for i in nb.prange(count):
        result[i] = np.random.poisson(5)

    return result


if __name__ == "__main__":
    unittest.main()

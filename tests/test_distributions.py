"""
Unit tests for laser.core.distributions module.

Note that this is not intended to test NumPy, Numba, or SciPy themselves, but rather to ensure that the
distributions implemented in laser.core.distributions have been "wired up" correctly.
"""

import sys
import unittest
from itertools import product
from time import perf_counter_ns

import numba as nb
import numpy as np
import pytest
from scipy.stats import beta as beta_ref
from scipy.stats import binom
from scipy.stats import expon
from scipy.stats import gamma as gamma_ref
from scipy.stats import ks_2samp
from scipy.stats import logistic as logistic_ref
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import uniform as uniform_ref
from scipy.stats import weibull_min

import laser.core.distributions as dists

NSAMPLES = 100_000
KS_THRESHOLD = 0.02  # Acceptable KS statistic for similarity


class TestDistributions(unittest.TestCase):
    def test_beta(self):
        params = [(0.5, 0.5), (5.0, 1.0), (1.0, 3.0), (2.0, 2.0), (2.0, 5.0)]
        for a, b in params:
            fn = dists.beta(a, b)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = beta_ref.rvs(a, b, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Beta({a},{b}) KS={stat}"

    def test_binomial(self):
        params = [(20, 0.5), (20, 0.7), (40, 0.5)]
        for n, p in params:
            fn = dists.binomial(n, p)
            samples = dists.sample_ints(fn, np.zeros(NSAMPLES, dtype=np.int32))
            ref_samples = binom.rvs(n, p, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Binomial({n},{p}) KS={stat}"

    def test_constant_float(self):
        values = [0.0, 1.0, -1.0, 3.14159, 2.71828]
        for value in values:
            fn = dists.constant_float(value)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            assert np.all(samples == np.float32(value)), f"Constant Float({value}) failed"

    def test_constant_int(self):
        values = [0, 1, -1, 42, 100]
        for value in values:
            fn = dists.constant_int(value)
            samples = dists.sample_ints(fn, np.zeros(NSAMPLES, dtype=np.int32))
            assert np.all(samples == np.int32(value)), f"Constant Int({value}) failed"

    def test_exponential(self):
        params = [0.5, 1.0, 1.5]
        for lam in params:
            scale = 1 / lam
            fn = dists.exponential(scale)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = expon.rvs(scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Exponential({scale}) KS={stat}"

    def test_gamma(self):
        params = [(1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (5.0, 1.0), (9.0, 0.5), (7.5, 1.0), (0.5, 1.0)]
        for shape, scale in params:
            fn = dists.gamma(shape, scale)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = gamma_ref.rvs(shape, scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Gamma({shape},{scale}) KS={stat}"

    def test_logistic(self):
        params = [(5, 2), (9, 3), (9, 4), (6, 2), (2, 1)]
        for loc, scale in params:
            fn = dists.logistic(loc, scale)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = logistic_ref.rvs(loc=loc, scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Logistic({loc},{scale}) KS={stat}"

    def test_lognormal(self):
        params = [(0, 1), (0, 0.5), (0, 0.25)]
        for mean, sigma in params:
            fn = dists.lognormal(mean, sigma)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = lognorm.rvs(sigma, scale=np.exp(mean), size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Lognormal({mean},{sigma}) KS={stat}"

    @unittest.skipIf(
        sys.version_info[:2] == (3, 10) and sys.platform == "darwin",
        "negative_binomial is flaky on macOS + Python 3.10 (see CHANGELOG.rst Unreleased)",
    )
    def test_negative_binomial(self):
        params = product([1, 2, 3, 4, 5], [1 / 2, 1 / 3, 1 / 4, 1 / 5])
        for r, p in params:
            fn = dists.negative_binomial(r, p)
            samples = dists.sample_ints(fn, np.zeros(NSAMPLES, dtype=np.int32))
            ref_samples = np.random.negative_binomial(r, p, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Negative Binomial({r},{p}) KS={stat}"

    def test_normal(self):
        params = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]
        for mu, sigmasq in params:
            sigma = np.sqrt(sigmasq)
            fn = dists.normal(mu, sigma)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = norm.rvs(loc=mu, scale=sigma, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Normal({mu},{sigma}) KS={stat}"

    def test_poisson(self):
        params = [1, 4, 10]
        for lam in params:
            fn = dists.poisson(lam)
            samples = dists.sample_ints(fn, np.zeros(NSAMPLES, dtype=np.int32))
            ref_samples = poisson.rvs(mu=lam, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Poisson({lam}) KS={stat}"

    def test_uniform(self):
        params = [(0.0, 1.0), (0.25, 1.25), (0.0, 2.0), (-1.0, 1.0), (2.71828, 3.14159), (1.30, 4.20)]
        for low, high in params:
            fn = dists.uniform(low, high)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = uniform_ref.rvs(loc=low, scale=high - low, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Uniform({low},{high}) KS={stat}"

    def test_weibull(self):
        params = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (5.0, 1.0)]
        for a, lam in params:
            fn = dists.weibull(a, lam)
            samples = dists.sample_floats(fn, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = weibull_min.rvs(a, scale=lam, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Weibull({a},{lam}) KS={stat}"

    @unittest.skip("Performance test, not a unit test")
    def test_perf(self):
        NSAMPLES = 1_000_000
        rng = np.random.default_rng(42)

        t0 = perf_counter_ns()
        _npsamples = rng.normal(loc=7.0, scale=0.875, size=NSAMPLES).astype(np.float32)
        t1 = perf_counter_ns()
        tnumpy = t1 - t0
        # print(f"NumPy normal: {tnumpy / 1_000_000:.2f} ms")
        # print(f"{_npsamples.mean():.4f} ± {_npsamples.std():.4f}")

        gaussian = dists.normal(loc=7.0, scale=0.875)
        _ = dists.sample_floats(gaussian, np.zeros(1000, dtype=np.float32))  # warmup
        t2 = perf_counter_ns()
        _nbsamples = dists.sample_floats(gaussian, np.zeros(NSAMPLES, dtype=np.float32))
        t3 = perf_counter_ns()
        tnumba = t3 - t2
        # print(f"LASER normal: {tnumba / 1_000_000:.2f} ms")
        # print(f"{_nbsamples.mean():.4f} ± {_nbsamples.std():.4f}")

        if nb.get_num_threads() > 2:
            assert (
                tnumba < tnumpy
            ), f"Numba-compatible distribution ({tnumba / 1_000_000:.2f} ms) slower than NumPy ({tnumpy / 1_000_000:.2f} ms)"


class TestCompositionHelpers(unittest.TestCase):
    """Tests for `mixture2`, `tick_modulated`, and `node_modulated`.

    These verify the composition factories return Numba-compatible samplers that
    produce the expected statistical behavior. Failure here means the composition
    pattern documented in the module is broken — either the closure doesn't compile
    or the resulting distribution doesn't match the analytic combination.
    """

    def test_mixture2_proportions_match_p_a(self):
        """Given a 2-component mixture of two constant samplers, when we draw many samples, then the proportion equal to value_a matches p_a within Monte Carlo tolerance."""
        const_a = dists.constant_float(1.0)
        const_b = dists.constant_float(2.0)
        p_a = 0.3
        sampler = dists.mixture2(const_a, const_b, p_a=p_a)
        out = dists.sample_floats(sampler, np.empty(NSAMPLES, dtype=np.float32))
        observed_p_a = float(np.mean(out == np.float32(1.0)))
        assert abs(observed_p_a - p_a) < 0.01, f"mixture2 proportion {observed_p_a} differs from p_a={p_a} by more than 1%"

    def test_mixture2_recovers_both_branches(self):
        """Given a mixture with p_a in (0,1), when we draw many samples, then both branches' values appear."""
        sampler = dists.mixture2(dists.constant_float(1.0), dists.constant_float(2.0), p_a=0.5)
        out = dists.sample_floats(sampler, np.empty(10_000, dtype=np.float32))
        unique = np.unique(out)
        assert set(unique.tolist()) == {1.0, 2.0}, f"mixture2 produced unexpected values: {unique}"

    def test_mixture2_rejects_invalid_p_a(self):
        """Given an out-of-range p_a, when mixture2 is called, then ValueError is raised."""
        with pytest.raises(ValueError, match=r"p_a must be in \[0, 1\]"):
            dists.mixture2(dists.constant_float(0.0), dists.constant_float(1.0), p_a=1.5)
        with pytest.raises(ValueError, match=r"p_a must be in \[0, 1\]"):
            dists.mixture2(dists.constant_float(0.0), dists.constant_float(1.0), p_a=-0.1)

    def test_tick_modulated_scales_by_modulator_at_tick(self):
        """Given a tick-modulated constant sampler, when we draw at a specific tick, then the output equals base_value * modulator[tick % L]."""
        modulator = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        sampler = dists.tick_modulated(dists.constant_float(10.0), modulator)
        for tick in range(8):  # cover two full periods
            out = dists.sample_floats(sampler, np.empty(100, dtype=np.float32), tick=tick)
            expected = np.float32(10.0 * modulator[tick % 4])
            assert np.allclose(out, expected), f"tick={tick}: got {out[0]}, expected {expected}"

    def test_tick_modulated_rejects_empty_or_2d_modulator(self):
        """Given an invalid modulator shape, when tick_modulated is called, then ValueError is raised."""
        with pytest.raises(ValueError, match=r"modulator must be a non-empty 1D array"):
            dists.tick_modulated(dists.constant_float(1.0), np.empty(0, dtype=np.float32))
        with pytest.raises(ValueError, match=r"modulator must be a non-empty 1D array"):
            dists.tick_modulated(dists.constant_float(1.0), np.ones((3, 3), dtype=np.float32))

    def test_node_modulated_scales_by_modulator_at_node(self):
        """Given a node-modulated constant sampler, when we draw at a specific node, then output equals base_value * modulator[node]."""
        modulator = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        sampler = dists.node_modulated(dists.constant_float(100.0), modulator)
        for node in range(5):
            out = dists.sample_floats(sampler, np.empty(100, dtype=np.float32), node=node)
            expected = np.float32(100.0 * modulator[node])
            assert np.allclose(out, expected), f"node={node}: got {out[0]}, expected {expected}"

    def test_node_modulated_rejects_empty_or_2d_modulator(self):
        """Given an invalid modulator shape, when node_modulated is called, then ValueError is raised."""
        with pytest.raises(ValueError, match=r"modulator must be a non-empty 1D array"):
            dists.node_modulated(dists.constant_float(1.0), np.empty(0, dtype=np.float32))
        with pytest.raises(ValueError, match=r"modulator must be a non-empty 1D array"):
            dists.node_modulated(dists.constant_float(1.0), np.ones((3, 3), dtype=np.float32))

    def test_composition_helpers_chain(self):
        """Given a node-modulated mixture, when we draw at a specific node, then the composition behaves as the product of the two effects."""
        # 50/50 mixture of 1.0 and 3.0 → average 2.0; node 2 multiplier 0.5 → average 1.0.
        mix = dists.mixture2(dists.constant_float(1.0), dists.constant_float(3.0), p_a=0.5)
        node_modulator = np.array([1.0, 1.0, 0.5, 1.0], dtype=np.float32)
        sampler = dists.node_modulated(mix, node_modulator)
        out = dists.sample_floats(sampler, np.empty(NSAMPLES, dtype=np.float32), node=2)
        # Expected values at node 2: 0.5 or 1.5 with equal probability → mean 1.0.
        assert abs(out.mean() - 1.0) < 0.02, f"chained composition mean {out.mean()} not close to 1.0"


class TestSampleConvenience(unittest.TestCase):
    """Tests for the `distributions.sample()` one-liner wrapper.

    These verify that `sample()` correctly (a) accepts a pre-built sampler and
    auto-allocates the output buffer, (b) accepts a factory + kwargs and builds
    the sampler on the fly, and (c) dispatches to `sample_floats` vs `sample_ints`
    based on dtype. Failure here means callers using the new ergonomic shortcut
    will get wrong dtypes, wrong shapes, or silently wrong distributions.
    """

    def test_sample_with_pre_built_sampler_floats(self):
        """Given a pre-built float sampler, when we call sample(fn, n), then we get a length-n float32 array filled by the sampler."""
        sampler = dists.constant_float(42.0)
        out = dists.sample(sampler, n=500)
        assert out.shape == (500,)
        assert out.dtype == np.float32
        assert np.all(out == np.float32(42.0))

    def test_sample_with_pre_built_sampler_ints(self):
        """Given a pre-built int sampler, when we call sample(fn, n), then we get a length-n int32 array filled by the sampler."""
        sampler = dists.constant_int(7)
        out = dists.sample(sampler, n=500)
        assert out.shape == (500,)
        assert out.dtype == np.int32
        assert np.all(out == np.int32(7))

    def test_sample_forwards_factory_kwargs(self):
        """Given a factory + kwargs, when we call sample(factory, n, **kwargs), then it builds the sampler on the fly and samples from it."""
        out = dists.sample(dists.uniform, n=NSAMPLES, low=10.0, high=11.0)
        assert out.dtype == np.float32
        assert np.all((out >= 10.0) & (out < 11.0))

    def test_sample_with_explicit_out_buffer(self):
        """Given an explicit out buffer, when we call sample(fn, ..., out=out), then n and dtype are taken from the buffer."""
        sampler = dists.constant_float(3.14)
        buf = np.empty(64, dtype=np.float32)
        result = dists.sample(sampler, n=999, out=buf)  # n is ignored
        assert result is buf
        assert result.shape == (64,)
        assert np.all(result == np.float32(3.14))

    def test_sample_with_explicit_dtype(self):
        """Given an explicit dtype, when we call sample(fn, n, dtype=dtype), then the output buffer uses that dtype."""
        sampler = dists.constant_float(2.5)
        out = dists.sample(sampler, n=100, dtype=np.float64)
        assert out.dtype == np.float64
        assert np.all(out == np.float64(2.5))

    def test_sample_rejects_non_numeric_dtype(self):
        """Given a non-numeric output dtype, when sample() runs, then TypeError is raised."""
        sampler = dists.constant_float(1.0)
        with pytest.raises(TypeError, match=r"output dtype must be integer or floating"):
            dists.sample(sampler, n=10, dtype=np.bool_)

    def test_sample_forwards_tick_and_node(self):
        """Given tick and node, when sample() runs with a tick-modulated sampler, then the per-tick multiplier is applied."""
        modulator = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        sampler = dists.tick_modulated(dists.constant_float(1.0), modulator)
        for tick in (0, 1, 2, 3):
            out = dists.sample(sampler, n=50, tick=tick)
            assert np.allclose(out, modulator[tick]), f"tick={tick}: got {out[0]}, expected {modulator[tick]}"


if __name__ == "__main__":
    unittest.main()

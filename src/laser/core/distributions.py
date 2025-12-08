"""
Various probability distributions implemented using NumPy and Numba.

LASER based models generally move from pure NumPy to Numba-accelerated
version of the core dynamics, e.g., transmission.

It would be a hassle to re-implement these functions for each desired
distribution, so we provide these Numba-wrapped distributions here which
can be passed in to other Numba compiled functions.

For example, a simple SIR model may want to parameterize the infectious
period using a distribution. By passing in a Numba-wrapped distribution
function, we can pick and parameterize a distribution based on
configuration and sample from that distribution within the Numba-compiled
SIR model without needing to re-implement the distribution logic within
the SIR model itself.

A simple example of usage::

    import laser.core.distributions as dist

    # Create a Numba-wrapped beta distribution
    beta_dist = dist.beta(2.0, 5.0)

    # Assign the distribution to the model's infectious period distribution
    # so the transmission component can sample from it during simulation
    model.infectious_period_dist = beta_dist

Note that the distribution functions take two parameters, ``tick`` and
``node``, which are currently unused but match the desired signature for
disease model components that may need to sample from distributions
based on the current simulation tick or node index. In other words,
distributions with spatial or temporal variation could be implemented in
the future.

Here are examples of Numba-wrapped distribution functions that could vary
based on tick or tick + node::

    # temporal variation only
    cosine = np.cos(np.linspace(0, np.pi * 2, 365))

    @nb.njit(nogil=True, cache=True)
    def seasonal_distribution(tick: int, node: int) -> np.float32:
        # ignore node for this seasonal distribution
        day_of_year = tick % 365
        base_value = 42.0 + 3.14159 * cosine[day_of_year]
        # parameterize normal() with seasonal factor
        return np.float32(np.random.normal(base_value, 2.0))

    # additional spatial variation
    ramp = np.linspace(0, 2, 42)

    @nb.njit(nogil=True, cache=True)
    def ramped_distribution(tick: int, node: int) -> np.float32:
        day_of_year = tick % 365
        base_value = 42.0 + 3.14159 * cosine[day_of_year]
        base_value *= ramp[node]
        return np.float32(np.random.normal(base_value, 1.0))

Normally, these distributions—built in or custom—will be used once per
agent as above. However, the ``sample_ints()`` and ``sample_floats()``
functions can be used to efficiently sample large arrays using multiple
CPU cores in parallel.
"""

from functools import lru_cache

import numba as nb
import numpy as np


# beta(a, b, size=None)
@lru_cache
def beta(a, b):
    r"""
    Beta distribution.
    $$f(x; a, b) = \\frac {x^{a-1} (1-x)^{b-1}} {B(a, b)}$$
    where B(a, b) is the beta function.
    """

    @nb.njit(nogil=True)
    def _beta(_tick: int, _node: int):
        return np.float32(np.random.beta(a, b))

    return _beta


# binomial(n, p, size=None)
@lru_cache
def binomial(n, p):
    r"""
    Binomial distribution.
    $$f(k,n,p) = Pr(X = k) = \\binom {n} {k} p^k (1-p)^{n-k}$$
    where *n* is the number of trials and *p* is the probability of success [0, 1].
    """

    @nb.njit(nogil=True)
    def _binomial(_tick: int, _node: int):
        return np.int32(np.random.binomial(n, p))

    return _binomial


@lru_cache
def constant_float(value):
    """
    Constant distribution.
    Always returns the same floating point value.
    """

    @nb.njit(nogil=True)
    def _constant(_tick: int, _node: int):
        return np.float32(value)

    return _constant


@lru_cache
def constant_int(value):
    """
    Constant distribution.
    Always returns the same integer value.
    """

    @nb.njit(nogil=True)
    def _constant(_tick: int, _node: int):
        return np.int32(value)

    return _constant


# exponential(scale=1.0, size=None)
@lru_cache
def exponential(scale):
    r"""
    Exponential distribution.
    $$f(x; \\frac {1} {\\beta}) = \\frac {1} {\\beta} e^{-\\frac {x} {\\beta}}$$
    where *β* is the scale parameter (β = 1 / λ).
    """

    @nb.njit(nogil=True)
    def _exponential(_tick: int, _node: int):
        return np.float32(np.random.exponential(scale))

    return _exponential


# gamma(shape, scale=1.0, size=None)
@lru_cache
def gamma(shape, scale):
    r"""
    Gamma distribution.
    $$p(x) = x^{k-1} \\frac {e^{- x / \\theta}}{\\theta^k \\Gamma(k)}$$
    where *k* is the shape, *θ* is the scale, and *Γ(k)* is the gamma function.
    """

    @nb.njit(nogil=True)
    def _gamma(_tick: int, _node: int):
        return np.float32(np.random.gamma(shape, scale))

    return _gamma


# logistic(loc=0.0, scale=1.0, size=None)
@lru_cache
def logistic(loc, scale):
    r"""
    Logistic distribution.
    $$P(x) = \\frac {e^{-(x - \\mu) / s}} {s (1 + e^{-(x - \\mu) / s})^2}$$
    where *μ* is the location parameter and *s* is the scale parameter.
    """

    @nb.njit(nogil=True)
    def _logistic(_tick: int, _node: int):
        return np.float32(np.random.logistic(loc, scale))

    return _logistic


# lognormal(mean=0.0, sigma=1.0, size=None)
@lru_cache
def lognormal(mean, sigma):
    r"""
    Log-normal distribution.
    $$p(x) = \\frac {1} {\\sigma x \\sqrt {2 \\pi}} e^{- \\frac {(\ln x - \\mu)^2} {2 \\sigma^2}}$$
    where *μ* is the mean and *σ* is the standard deviation of the underlying normal distribution.
    """

    @nb.njit(nogil=True)
    def _lognormal(_tick: int, _node: int):
        return np.float32(np.random.lognormal(mean, sigma))

    return _lognormal


# # multinomial(n, pvals, size=None)
# @lru_cache
# def multinomial(n, pvals):
#     @nb.njit(nogil=True)
#     def _multinomial():
#         return np.int32(np.random.multinomial(n, pvals))
#
#     return _multinomial


# negative_binomial(n, p, size=None)
@lru_cache
def negative_binomial(n, p):
    r"""
    Negative binomial distribution.
    $$P(N; n, p) = \\frac {\\Gamma (N + n)} {N! \\Gamma (n)} p^n (1 - p)^N$$
    where *n* is the number of successes, *p* is the probability of success on each trial, *N + n* is the number of trials, and *Γ()* is the gamma function.
    When *n* is an integer,
    $$\\frac {\\Gamma (N + n)} {N! \\Gamma (n)} = \\binom {N + n - 1} {n - 1}$$
    which is the more common form of this term.
    """

    @nb.njit(nogil=True)
    def _negative_binomial(_tick: int, _node: int):
        return np.int32(np.random.negative_binomial(n, p))

    return _negative_binomial


# normal(loc=0.0, scale=1.0, size=None)
@lru_cache
def normal(loc, scale):
    r"""
    Normal (Gaussian) distribution.
    $$p(x) = \\frac {1} {\\sqrt {2 \\pi \\sigma^2}} e^{- \\frac {(x - \\mu)^2} {2 \\sigma^2}}$$
    where *μ* is the mean and *σ* is the standard deviation.
    """

    @nb.njit(nogil=True)
    def _normal(_tick: int, _node: int):
        return np.float32(np.random.normal(loc, scale))

    return _normal


# poisson(lam=1.0, size=None)
@lru_cache
def poisson(lam):
    r"""
    Poisson distribution.
    $$f( k ; \\lambda ) = \\frac {\\lambda^k e^{- \\lambda}} {k!}$$
    where *λ* is the expected number of events in the given interval.
    """

    @nb.njit(nogil=True)
    def _poisson(_tick: int, _node: int):
        return np.int32(np.random.poisson(lam))

    return _poisson


# uniform(low=0.0, high=1.0, size=None)
@lru_cache
def uniform(low, high):
    r"""
    Uniform distribution.
    $$p(x) = \\frac {1} {b - a}$$
    where *a* is the lower bound and *b* is the upper bound, [*a*, *b*).
    """

    @nb.njit(nogil=True)
    def _uniform(_tick: int, _node: int):
        return np.float32(np.random.uniform(low, high))

    return _uniform


# weibull(a, size=None)
@lru_cache
def weibull(a, lam):
    r"""
    Weibull distribution.
    $$X = \\lambda (- \\ln ( U ))^{1 / a}$$
    where *a* is the shape parameter and *λ* is the scale parameter.
    """

    @nb.njit(nogil=True)
    def _weibull(_tick: int, _node: int):
        return np.float32(lam * np.random.weibull(a))

    return _weibull


# Shared Numba sampling functions
@nb.njit(parallel=True, nogil=True)
def sample_floats(fn, dest, tick=0, node=0):
    """
    Fill an array with floating point values sampled from a Numba-wrapped distribution function.

    Parameters
    ----------
    fn : function
        Numba-wrapped distribution function returning float32 values.
    dest : np.ndarray
        Pre-allocated destination float32 array to store samples.
    tick : int, optional
        Current simulation tick (default is 0). Passed through to the distribution function.
    node : int, optional
        Current node index (default is 0). Passed through to the distribution function.

    Returns
    -------
    np.ndarray
        The destination array filled with sampled values.
    """
    count = dest.shape[0]
    for i in nb.prange(count):
        dest[i] = fn(tick, node)
    return dest


@nb.njit(parallel=True, nogil=True)
def sample_ints(fn, dest, tick=0, node=0):
    """
    Fill an array with integer values sampled from a Numba-wrapped distribution function.

    Parameters
    ----------
    fn : function
        Numba-wrapped distribution function returning int32 values.
    dest : np.ndarray
        Pre-allocated destination int32 array to store samples.
    tick : int, optional
        Current simulation tick (default is 0). Passed through to the distribution function.
    node : int, optional
        Current node index (default is 0). Passed through to the distribution function.

    Returns
    -------
    np.ndarray
        The destination array filled with sampled values.
    """
    count = dest.shape[0]
    for i in nb.prange(count):
        dest[i] = fn(tick, node)
    return dest

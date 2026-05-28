"""
Various probability distributions implemented using NumPy and Numba.

LASER based models generally move from pure NumPy to Numba-accelerated version of the core dynamics, e.g., transmission.

It would be a hassle to re-implement these functions for each desired distribution, so we provide these Numba-wrapped distributions here which can be passed in to other Numba compiled functions.

For example, a simple SIR model may want to parameterize the infectious period using a distribution. By passing in a Numba-wrapped distribution function, we can pick and parameterize a distribution based on configuration and sample from that distribution within the Numba-compiled SIR model without needing to re-implement the distribution logic within the SIR model itself.

A simple example of usage::

    import laser.core.distributions as dist

    # Create a Numba-wrapped beta distribution
    beta_dist = dist.beta(2.0, 5.0)

    # Assign the distribution to the model's infectious period distribution
    # so the transmission component can sample from it during simulation
    model.infectious_period_dist = beta_dist

Note that the distribution functions take two parameters, ``tick`` and ``node``, which are currently unused but match the desired signature for disease model components that may need to sample from distributions based on the current simulation tick or node index. In other words, distributions with spatial or temporal variation could be implemented in the future.

Here are examples of Numba-wrapped distribution functions that could vary based on tick or tick + node::

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
        # use seasonal factor
        base_value = 42.0 + 3.14159 * cosine[day_of_year]
        # apply spatial ramp based on node index
        base_value *= ramp[node]
        # parameterize normal() with seasonal + spatial factor
        return np.float32(np.random.normal(base_value, 1.0))

Normally, these distributions—built in or custom—will be used once per agent as above. However, the ``sample_ints()`` and ``sample_floats()`` functions can be used to efficiently sample large arrays using multiple CPU cores in parallel.
"""

from functools import lru_cache

import numba as nb
import numpy as np


# beta(a, b, size=None)
@lru_cache
def beta(a, b):
    r"""
    Beta distribution.

    $f(x; a, b) = \frac {x^{a-1} (1-x)^{b-1}} {B(a, b)}$

    where $B(a, b)$ is the beta function.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.beta(2.0, 5.0)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    $f(k,n,p) = Pr(X = k) = \binom {n} {k} p^k (1-p)^{n-k}$

    where $n$ is the number of trials and $p$ is the probability of success [0, 1].

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.binomial(10, 0.3)
        out = np.empty(1_000, dtype=np.int32)
        dist.sample_ints(sampler, out)
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

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.constant_float(0.42)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.constant_int(7)
        out = np.empty(1_000, dtype=np.int32)
        dist.sample_ints(sampler, out)
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

    $f(x; \frac {1} {\beta}) = \frac {1} {\beta} e^{-\frac {x} {\beta}}$

    where $\beta$ is the scale parameter ($\beta = 1 / \lambda$).

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.exponential(scale=3.5)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    $p(x) = x^{k-1} \frac {e^{- x / \theta}}{\theta^k \Gamma(k)}$

    where $k$ is the shape, $\theta$ is the scale, and $\Gamma(k)$ is the gamma function.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.gamma(shape=2.0, scale=1.5)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    $P(x) = \frac {e^{-(x - \mu) / s}} {s (1 + e^{-(x - \mu) / s})^2}$

    where $\mu$ is the location parameter and $s$ is the scale parameter.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.logistic(loc=0.0, scale=1.0)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    $p(x) = \frac {1} {\sigma x \sqrt {2 \pi}} e^{- \frac {(\ln x - \mu)^2} {2 \sigma^2}}$

    where $\mu$ is the mean and $\sigma$ is the standard deviation of the underlying normal distribution.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.lognormal(mean=0.0, sigma=0.5)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    $P(N; n, p) = \frac {\Gamma (N + n)} {N! \Gamma (n)} p^n (1 - p)^N$

    where $n$ is the number of successes, $p$ is the probability of success on each trial, $N + n$ is the number of trials, and $\Gamma()$ is the gamma function.
    When $n$ is an integer,

    $\frac {\Gamma (N + n)} {N! \Gamma (n)} = \binom {N + n - 1} {n - 1}$

    which is the more common form of this term.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.negative_binomial(n=5, p=0.3)
        out = np.empty(1_000, dtype=np.int32)
        dist.sample_ints(sampler, out)
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

    $p(x) = \frac {1} {\sqrt {2 \pi \sigma^2}} e^{- \frac {(x - \mu)^2} {2 \sigma^2}}$

    where $\mu$ is the mean and $\sigma$ is the standard deviation.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.normal(loc=0.0, scale=1.0)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    $f( k ; \lambda ) = \frac {\lambda^k e^{- \lambda}} {k!}$

    where $\lambda$ is the expected number of events in the given interval.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.poisson(lam=3.0)
        out = np.empty(1_000, dtype=np.int32)
        dist.sample_ints(sampler, out)
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

    $p(x) = \frac {1} {b - a}$

    where $a$ is the lower bound and $b$ is the upper bound, [$a$, $b$).

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.uniform(low=0.0, high=1.0)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    $X = \lambda (- \ln ( U ))^{1 / a}$

    where $a$ is the shape parameter and $\lambda$ is the scale parameter.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        sampler = dist.weibull(a=1.5, lam=2.0)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
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

    Args:
        fn (function): Numba-wrapped distribution function returning float32 values.
        dest (np.ndarray): Pre-allocated destination float32 array to store samples.
        tick (int, optional): Current simulation tick (default is 0). Passed through to the distribution function.
        node (int, optional): Current node index (default is 0). Passed through to the distribution function.

    Returns:
        dest (np.ndarray): The destination array filled with sampled values.
    """
    count = dest.shape[0]
    for i in nb.prange(count):
        dest[i] = fn(tick, node)
    return dest


@nb.njit(parallel=True, nogil=True)
def sample_ints(fn, dest, tick=0, node=0):
    """
    Fill an array with integer values sampled from a Numba-wrapped distribution function.

    Args:
        fn (function): Numba-wrapped distribution function returning int32 values.
        dest (np.ndarray): Pre-allocated destination int32 array to store samples.
        tick (int, optional): Current simulation tick (default is 0). Passed through to the distribution function.
        node (int, optional): Current node index (default is 0). Passed through to the distribution function.

    Returns:
        dest (np.ndarray): The destination array filled with sampled values.
    """
    count = dest.shape[0]
    for i in nb.prange(count):
        dest[i] = fn(tick, node)
    return dest


# Composition helpers — productize the patterns shown in the module-level docstring.
# These all take and return Numba-wrapped samplers with the same `(tick, node) -> scalar`
# signature, so they compose freely and are valid `fn` arguments to `sample_floats` /
# `sample_ints`. They are float-only (returning `float32`); int samplers should be
# composed at the Python level or wrapped in `np.int32(...)` by the caller.


def mixture2(sampler_a, sampler_b, p_a):
    """Two-component mixture sampler.

    With probability ``p_a`` returns a draw from ``sampler_a``; otherwise returns a draw
    from ``sampler_b``. Both samplers must be Numba-wrapped distribution functions
    returning ``float32``.

    Args:
        sampler_a: Numba-wrapped sampler chosen with probability ``p_a``.
        sampler_b: Numba-wrapped sampler chosen with probability ``1 - p_a``.
        p_a (float): Mixture weight on ``sampler_a``, in [0, 1].

    Returns:
        function: A Numba-wrapped sampler with signature ``(tick, node) -> float32``.

    Raises:
        ValueError: If ``p_a`` is not in [0, 1].

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        short = dist.exponential(scale=3.0)
        long = dist.exponential(scale=14.0)
        # 70% of agents recover with the short timescale, 30% with the long timescale.
        sampler = dist.mixture2(short, long, p_a=0.7)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out)
    """
    if not (0.0 <= p_a <= 1.0):
        raise ValueError(f"p_a must be in [0, 1] (got {p_a!r})")
    threshold = np.float32(p_a)

    @nb.njit(nogil=True)
    def _mixture2(_tick: int, _node: int):
        if np.random.random() < threshold:
            return np.float32(sampler_a(_tick, _node))
        return np.float32(sampler_b(_tick, _node))

    return _mixture2


def tick_modulated(base_sampler, modulator):
    """Scale a sampler's output by a tick-periodic multiplier.

    At each call the output of ``base_sampler(tick, node)`` is multiplied by
    ``modulator[tick % len(modulator)]``. The common case is a seasonal multiplier
    of length 365 driving annual variation in an underlying distribution.

    Args:
        base_sampler: Numba-wrapped sampler with signature ``(tick, node) -> scalar``.
        modulator (array-like of float): 1D array of per-tick multipliers. Cast to
            ``float32`` internally; length sets the period.

    Returns:
        function: A Numba-wrapped sampler with signature ``(tick, node) -> float32``.

    Raises:
        ValueError: If ``modulator`` is not a non-empty 1D array.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        seasonal = 1.0 + 0.3 * np.cos(np.linspace(0, 2 * np.pi, 365))
        base = dist.normal(loc=10.0, scale=1.0)
        sampler = dist.tick_modulated(base, seasonal)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out, tick=42)
    """
    modulator = np.asarray(modulator, dtype=np.float32)
    if modulator.ndim != 1 or modulator.size == 0:
        raise ValueError(f"modulator must be a non-empty 1D array (got shape {modulator.shape})")
    period = modulator.size

    @nb.njit(nogil=True)
    def _tick_modulated(_tick: int, _node: int):
        return np.float32(modulator[_tick % period] * base_sampler(_tick, _node))

    return _tick_modulated


def node_modulated(base_sampler, modulator):
    """Scale a sampler's output by a per-node multiplier.

    At each call the output of ``base_sampler(tick, node)`` is multiplied by
    ``modulator[node]``. Useful for per-patch transmission ramps or spatial
    heterogeneity in any distribution parameter.

    Args:
        base_sampler: Numba-wrapped sampler with signature ``(tick, node) -> scalar``.
        modulator (array-like of float): 1D array of per-node multipliers. Cast to
            ``float32`` internally; ``node`` arguments must be valid indices into it.

    Returns:
        function: A Numba-wrapped sampler with signature ``(tick, node) -> float32``.

    Raises:
        ValueError: If ``modulator`` is not a non-empty 1D array.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        ramp = np.linspace(0.5, 2.0, 42, dtype=np.float32)
        base = dist.normal(loc=10.0, scale=1.0)
        sampler = dist.node_modulated(base, ramp)
        out = np.empty(1_000, dtype=np.float32)
        dist.sample_floats(sampler, out, node=7)
    """
    modulator = np.asarray(modulator, dtype=np.float32)
    if modulator.ndim != 1 or modulator.size == 0:
        raise ValueError(f"modulator must be a non-empty 1D array (got shape {modulator.shape})")

    @nb.njit(nogil=True)
    def _node_modulated(_tick: int, _node: int):
        return np.float32(modulator[_node] * base_sampler(_tick, _node))

    return _node_modulated


def sample(fn_or_factory, n, *, dtype=None, tick=0, node=0, out=None, **factory_kwargs):
    """One-liner sampler: allocate a buffer and dispatch to the right `sample_*` kernel.

    Wraps the typical two-step "build sampler, allocate buffer, call `sample_floats` /
    `sample_ints`" pattern into a single call. Also forwards `**factory_kwargs` to
    `fn_or_factory` when it is a factory (e.g. [`normal`][laser.core.distributions.normal])
    rather than a pre-built sampler, so per-call distribution parameters do not need to be
    baked into the sampler factory ahead of time.

    Args:
        fn_or_factory: Either a Numba-wrapped sampler `(tick, node) -> scalar`, OR a factory
            function (e.g. [`normal`][laser.core.distributions.normal]) that takes
            `**factory_kwargs` and returns such a sampler.
        n (int): Number of samples to draw. Ignored when `out` is provided.
        dtype (np.dtype, optional): Output array dtype. Inferred from a probe call to the
            sampler when omitted. Must be either an integer or floating dtype.
        tick (int, optional): Simulation tick forwarded to the sampler (default `0`).
        node (int, optional): Node index forwarded to the sampler (default `0`).
        out (np.ndarray, optional): Pre-allocated output buffer. If provided, `n` and
            `dtype` are ignored and the buffer's shape/dtype drive the call.
        **factory_kwargs: Additional keyword arguments forwarded to `fn_or_factory` if and
            only if `factory_kwargs` is non-empty. The result of that call is then used as
            the sampler. Ignored otherwise.

    Returns:
        np.ndarray: The output array, filled in place.

    Raises:
        TypeError: If the resolved dtype is neither integer nor floating.

    **Example**:

        import numpy as np
        from laser.core import distributions as dist

        # One-liner: factory + sampling in a single call.
        out = dist.sample(dist.normal, n=1_000, loc=0.0, scale=1.0)

        # Or pass a pre-built sampler when you want to reuse it.
        gaussian = dist.normal(loc=0.0, scale=1.0)
        out = dist.sample(gaussian, n=1_000)
    """
    fn = fn_or_factory(**factory_kwargs) if factory_kwargs else fn_or_factory

    if out is None:
        if dtype is None:
            # Probe the sampler once to infer integer vs. floating output. Numba
            # auto-unboxes the return value to a Python `int`/`float`, so `np.asarray`
            # alone would report `int64`/`float64`; we want to match the project-wide
            # int32/float32 convention used by the distribution factories.
            probe = fn(int(tick), int(node))
            dtype = np.int32 if isinstance(probe, int | np.integer) else np.float32
        out = np.empty(int(n), dtype=dtype)

    if np.issubdtype(out.dtype, np.integer):
        return sample_ints(fn, out, tick=tick, node=node)
    if np.issubdtype(out.dtype, np.floating):
        return sample_floats(fn, out, tick=tick, node=node)
    raise TypeError(f"sample() output dtype must be integer or floating (got {out.dtype})")

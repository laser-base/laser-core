=====
Usage
=====

This page is a quick-start tour of the public surface of ``laser.core``. For
end-to-end examples, see :doc:`example`, :doc:`vdexample`, and
:doc:`spatialexample`.

Importing the package
=====================

The top-level package re-exports the three core classes and the most useful
submodules so that everything you need is one short import away::

    import laser.core as lc

    # core classes
    lc.LaserFrame
    lc.PropertySet
    lc.SortedQueue

    # submodules
    lc.distributions   # Numba-accelerated samplers (constant, uniform, poisson, ...)
    lc.migration       # gravity / radiation / Stouffer / competing-destinations
    lc.random          # process-wide seedable PRNG
    lc.demographics    # population pyramids and Kaplan-Meier estimator
    lc.utils           # capacity estimation, spatial grids

Setting the PRNG seed
=====================

``laser.core`` uses a single process-wide PRNG so that simulations are
reproducible regardless of which submodule consumes random draws::

    from laser.core import random

    random.seed(20260514)
    rng = random.prng()

Allocating an agent frame
=========================

:class:`LaserFrame <laser.core.LaserFrame>` is the agent / patch container. It
behaves like a fixed-capacity, dynamically-grown columnar table::

    import numpy as np
    from laser.core import LaserFrame

    frame = LaserFrame(capacity=10_000)
    frame.add_scalar_property("age", dtype=np.int32, default=0)
    frame.add_scalar_property("infected", dtype=bool, default=False)
    frame.add_vector_property("position", length=2, dtype=np.float32, default=0.0)

    start, end = frame.add(100)              # activate 100 agents
    frame.age[start:end] = np.arange(100)    # in-place writes

Composing parameter sets
========================

:class:`PropertySet <laser.core.PropertySet>` is a lightweight bag of named
parameters that composes with ``+=``, ``<<=``, and ``|=``::

    from laser.core import PropertySet

    base = PropertySet({"beta": 0.4, "gamma": 0.1})
    override = PropertySet({"beta": 0.6})
    base |= override          # right-hand wins for keys present in both

Sampling from a distribution
============================

:mod:`laser.core.distributions` exposes Numba-compatible distribution
*factories*: call one with the distribution's parameters to build a sampler,
then either invoke it once per agent or fill an array in bulk with
:func:`~laser.core.distributions.sample_floats` /
:func:`~laser.core.distributions.sample_ints` (or the
:func:`~laser.core.distributions.sample` one-liner)::

    import numpy as np
    from laser.core import distributions as dist

    # Pattern 1 — build the sampler, then fill a pre-allocated buffer.
    poisson_sampler = dist.poisson(lam=3.0)
    out = np.empty(1000, dtype=np.int32)
    dist.sample_ints(poisson_sampler, out)

    # Pattern 2 — one-liner: factory + sampling in a single call. `sample`
    # forwards `**factory_kwargs` to the factory, allocates the output buffer,
    # and dispatches to `sample_ints`/`sample_floats` based on dtype.
    out = dist.sample(dist.poisson, n=1000, lam=3.0)

    # The same call shape works for any of the distribution factories:
    out = dist.sample(dist.normal, n=1000, loc=0.0, scale=1.0)

Building a migration network
============================

Use :mod:`laser.core.migration` to construct migration matrices from any of
the supported spatial-interaction models::

    import numpy as np
    from laser.core import migration

    pops = np.array([10_000, 5_000, 2_000], dtype=np.float64)
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 1.0, 2.0])
    distances = migration.distance(lat, lon)

    # Run a model and row-normalize in one call:
    network = migration.build_network(
        migration.gravity, pops, distances,
        max_rowsum=0.05, k=0.01, a=1.0, b=1.0, c=2.0,
    )

    # Equivalent two-step form:
    # network = migration.gravity(pops, distances, k=0.01, a=1.0, b=1.0, c=2.0)
    # network = migration.row_normalizer(network, max_rowsum=0.05)

Next steps
==========

* :doc:`example` — full SIR walkthrough.
* :doc:`vdexample` — vital-dynamics example.
* :doc:`spatialexample` — spatial / migration example.
* :doc:`migration` — migration model reference.
* :doc:`kmestimator` — Kaplan-Meier age-at-death reference.
* :doc:`performance` — performance and parallelization guidance.

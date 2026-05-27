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

Use the Numba-accelerated samplers in :mod:`laser.core.distributions`::

    from laser.core import distributions, random

    rng = random.prng()
    n = 1000
    out = np.empty(n, dtype=np.float32)
    distributions.poisson(rng, lam=3.0, out=out)

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

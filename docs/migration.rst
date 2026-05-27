=========
Migration
=========

Some things to note:

These models all take in some parameters, along with a vector of populations and distances between nodes, and spit out a matrix defining the connection between nodes.

- When comparing to other models, it's important to consider how implementation of the migration / connection model affects interpretation of the parameters.  For example, a migration matrix could be implemented as a per-capita rate of travel from :math:`i` to :math:`j`, or as a total flux of people from :math:`i` to :math:`j`.  If your migration model has a term that scales like :math:`p_i^a`, using a per-capita rate introduces an implicit :math:`+1` into the exponent.  Or, depending on how infectivity / mixing is handled locally, if the introduced infectivity is normalized to local population, that might introduce an ambiguity in interpreting the exponent on the destination population, is it :math:`b` or :math:`b-1`, effectively?  In the end, these are terms that could be calibrated away, but useful to keep this in mind for interpretation and comparison against other models - we aim to support users defining their own migration models and implementations, and this sort of ambiguity is important to keep in mind.

- I'm aiming to do most of this with element-by-element numpy functions when I can, though loops would probably translate more obviously to numba/c.  I don't expect that computation of these matrices will be a substantial part of overall computational spend on a model either way.

- I have not tested nor written code to enforce conditions on the inputs.  This should be done - e.g., populations coming in as integers can present wrapping issues when we start exponentiating and multiplying them (signed integers in particular can be a problem because you may wrap into negative numbers).  So some input checking and such needs to be done.

- It's also worth investigating whether using large floats makes sense when computing these formulas, or whether we should put operations in a specific order.  This is because depending on the choice of someone's metapop network and spatial model parameters, and the order of computations, we can end up multiplying, dividing, summing over numbers that can be across really different scales, so weirdness might happen with loss of precision?  The integer issue above is more concerning and one that I have run into in the past.

- Distances on the diagonal of the distance matrix should always be 0.  We should check for 0s elsewhere and throw an error.  It's also nice to be able to use numpy element-by-element math without constant div-by-zero errors for the diagonal elements, so maybe each function should start by adding epsilon to the diagonal of the distance matrix?  We're going to zero out those terms in the network anyway...

.. _choosing-a-model:

Choosing a model
================

The four migration models in :mod:`laser.core.migration` produce the same kind of
output — an :math:`N \times N` flow matrix — from the same kind of inputs, but
they encode different assumptions about *what drives migration*. This section is
descriptive rather than prescriptive: laser-core is an unopinionated primitive
set, and the right model for a given study is a modeling decision, not a
software decision.

That said, **the gravity model is the natural starting point**. It is the most
widely-studied form, every other model in this module can be motivated as a
modification of it, and its parameters (population exponents and a distance
falloff) match the levers most modelers reach for first. In the absence of a
specific reason to prefer otherwise, starting with ``gravity`` and adjusting
parameters from there respects the principle of least surprise — both for
readers of the model code and for collaborators who have to interpret the
calibration.

What each model emphasizes
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Model
     - What it encodes
   * - :func:`~laser.core.migration.gravity`
     - Flow scales as :math:`p_i^a \cdot p_j^b / d_{i,j}^c`. The standard
       physics analogy — origin "push," destination "pull," distance penalty.
       Three free parameters in addition to the overall scale, all
       independently calibratable. Reasonable starting point when nothing else
       constrains the choice.
   * - :func:`~laser.core.migration.competing_destinations`
     - Gravity plus an explicit term for the influence of *other* attractive
       destinations near :math:`j`. Useful when the modeler suspects that a
       nearby alternative destination boosts (positive :math:`\delta`) or
       suppresses (negative :math:`\delta`) flows to :math:`j` — i.e. when
       agglomeration or screening effects matter.
   * - :func:`~laser.core.migration.stouffer`
     - Distance is replaced by the *number of intervening opportunities*
       (population at locations closer than :math:`j`). Best fit when the
       conceptual model is "people travel to the nearest attractive option,"
       e.g. commuting, job search, market access.
   * - :func:`~laser.core.migration.radiation`
     - A parameter-free formulation (only an overall scale :math:`k`) derived
       from the same intervening-opportunities intuition as Stouffer but with
       a specific functional form. Useful when calibration data is scarce and
       a "let the populations and geography speak" baseline is wanted.

Parameter intuitions
--------------------

These are descriptive heuristics, not recommended values; reach for the cited
literature for any production calibration.

- :math:`k` (all models) — overall scale. Sets total migration flux; usually
  calibrated against a target out-migration rate or per-capita travel volume.
- :math:`a, b` (gravity, competing-destinations, Stouffer) — exponents on the
  origin and destination populations. Values near :math:`1.0` are the standard
  starting point; departures from :math:`1.0` capture nonlinear scaling of
  flows with population. Note the :math:`\pm 1` ambiguity discussed in the
  intro depending on how local mixing and per-capita normalization are
  modeled.
- :math:`c` (gravity, competing-destinations) — distance exponent. Larger
  :math:`c` localizes flows; smaller :math:`c` spreads them. Empirically often
  in the range :math:`1 \le c \le 3` for human mobility.
- :math:`\delta` (competing-destinations) — sign matters: positive for
  agglomeration / synergy among nearby destinations, negative for screening /
  antagonism. Zero recovers the gravity model.
- ``include_home`` (Stouffer, radiation) — whether the origin's own population
  is counted in the "intervening opportunities" set :math:`\Omega(i,j)`. This
  is a modeling choice about whether local mixing competes with outbound
  travel; both are defensible and the right answer depends on what is being
  modeled.

Numerical edge cases to watch for
---------------------------------

- **Integer populations.** Exponentiating integer populations (especially
  signed 32-bit) can wrap into negative numbers. Cast to ``np.float64`` before
  passing :math:`p` to a migration model, or use unsigned dtypes large enough
  to hold :math:`\max p^a`.
- **Zero distances off the diagonal.** All four models treat the diagonal as
  the source node and zero it on output, but a zero distance between two
  *different* nodes will produce :math:`\inf` in gravity / competing-destinations
  (division by :math:`d^c`) and ill-defined ranks in Stouffer / radiation.
  Deduplicate or perturb coincident coordinates before calling.
- **Equidistant destinations.** All ties in distance are handled correctly
  (see the regression tests in ``tests/test_migration.py``): every member of a
  tied group receives the same intervening-population sum. Modelers who
  expected ties to be tie-broken arbitrarily should be aware that this is the
  intended behavior.
- **Output normalization.** None of the models return a probability
  distribution; they return a flux matrix. Use
  :func:`~laser.core.migration.row_normalizer` (or
  :func:`~laser.core.migration.build_network` with ``max_rowsum=...``) to cap
  per-node outflow when interpreting the network as a per-timestep migration
  rate.

Gravity model
=============

Functional form:
:math:`M_{i,j} = k \frac{p_i^a p_j^b}{d_{i,j}^c}`

Special cases of the gravity model (as noted above, both population exponents are subject to :math:`\pm1` ambiguity depending on implementation of spatial connectivity and local mixing):

- Xia's model: :math:`a = 0`
- mean-field model: :math:`c = 0, a = 1, b = 1`
- spatial diffusion: :math:`a = 0, b = 0`

Example usage
-------------

Below is an example of how to use the gravity model to compute migration flows between populations located at specific distances. The example assumes unequal population sizes and calculates the number of migrants moving between nodes based on the gravity model.

.. code-block:: python

    import numpy as np
    from laser.core.migration import gravity

    # Define populations and distances
    populations = np.array([5000, 10000, 15000, 20000, 25000])  # Unequal populations
    distances = np.array([
        [0.0, 10.0, 15.0, 20.0, 25.0],
        [10.0, 0.0, 10.0, 15.0, 20.0],
        [15.0, 10.0, 0.0, 10.0, 15.0],
        [20.0, 15.0, 10.0, 0.0, 10.0],
        [25.0, 20.0, 15.0, 10.0, 0.0]
    ])

    # Gravity model parameters
    k = 0.1    # Scaling constant
    a = 0.5    # Exponent for the population of the origin
    b = 1.0    # Exponent for the population of the destination
    c = 2.0    # Exponent for the distance

    # Compute the gravity model network
    migration_network = gravity(populations, distances, k=k, a=a, b=b, c=c)

    # Normalize to ensure total migrations represent 1% of the population
    total_population = np.sum(populations)
    migration_fraction = 0.01  # 1% of the population migrates
    scaling_factor = (total_population * migration_fraction) / np.sum(migration_network)
    migration_network *= scaling_factor

    # Generate a node ID array for agents
    node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(populations)])

    # Initialize a 2D array for migration counts
    migration_matrix = np.zeros_like(distances, dtype=int)

    # Select migrants based on the gravity model
    for origin in range(len(populations)):
        for destination in range(len(populations)):
            if origin != destination:
                # Number of migrants to move from origin to destination
                num_migrants = int(migration_network[origin, destination])
                # Select migrants randomly
                origin_ids = np.where(node_ids == origin)[0]
                selected_migrants = np.random.choice(origin_ids, size=num_migrants, replace=False)
                # Update the migration matrix
                migration_matrix[origin, destination] = num_migrants

This example demonstrates the end-to-end process of using the gravity model to calculate migration flows and randomly assign agents to those flows. The resulting migration matrix shows the number of individuals migrating between nodes.

Capping the total fraction of population that can migrate / infectivity that can be exported on a given timestep
================================================================================================================

Because the inputs to spatial models (populations, distances) can vary over many orders of magnitude, we can run into situations where a small number of nodes, often those closest to but distinct from large population centers, will end up with huge outflows. The :func:`laser.core.migration.build_network` helper composes the two steps — run the chosen model, then row-normalize so no node's total outflow exceeds the cap — into a single call:

.. code-block:: python

    import numpy as np
    from laser.core.migration import build_network, gravity

    pops = np.array([5000, 10000, 15000, 20000, 25000], dtype=np.float64)
    distances = np.array([
        [0.0, 10.0, 15.0, 20.0, 25.0],
        [10.0, 0.0, 10.0, 15.0, 20.0],
        [15.0, 10.0, 0.0, 10.0, 15.0],
        [20.0, 15.0, 10.0, 0.0, 10.0],
        [25.0, 20.0, 15.0, 10.0, 0.0],
    ])

    # Cap total outflow at 1% of source population per node, per timestep.
    network = build_network(
        gravity,
        pops,
        distances,
        max_rowsum=0.01,
        k=0.1, a=0.5, b=1.0, c=2.0,
    )

The same call shape works with :func:`~laser.core.migration.radiation`, :func:`~laser.core.migration.stouffer`, and :func:`~laser.core.migration.competing_destinations` — pass any of them as the first argument and supply that model's parameters as keyword arguments. Omitting ``max_rowsum`` (the default) returns the raw model output without normalization, equivalent to calling the model function directly.

The Competing Destinations model
================================

There are many models that aim to account for the impact of competition or synergy between potential destinations. Some aim to account for some "screening" effect of travel to distant destinations due to competition from attractive destinations closer to the origin :math:`i`. This model, in contrast, (Fotheringham AS. Spatial flows and spatial patterns. Environment and Planning A. 1984;16(4):529–543) aims to account for effects from other attractive destinations near destination :math:`j`; notably, this effect could be synergistic or antagonistic, depending on the sign of the exponent :math:`\delta`.

For example, in a "synergistic" version, perhaps migratory flow from Boston to Baltimore is higher than flow between two comparator cities of similar population and at similar distance, because the proximity of Washington, D.C. to Baltimore makes travel to Baltimore more attractive to Bostonians – this would be accounted for by a positive value of :math:`\delta`. On the other hand, this term may also be "antagonistic", if Washington is such an attractive destination that Bostonians eschew travel to Baltimore entirely; this would indicate a negative value of :math:`\delta`.

Mathematical Formulation:
:math:`M_{i,j} = k \frac{p_i^a p_j^b}{d_{i,j}^c} \left(\sum_{k \ne i,j} \frac{p_k^b}{d_{jk}^c}\right)^\delta`

Stouffer's rank model
=====================

Stouffer (Stouffer SA. Intervening opportunities: a theory relating mobility and distance. American Sociological Review. 1940;5(6):845–867) argued that human mobility patterns do not respond to absolute distance directly, but only indirectly through the accumulation of intervening opportunities for destinations. Stouffer thus proposed a model with no distance-dependence at all, rather only a term that accounts for all potential destinations closer than destination :math:`j`; thus, longer-distance travel depends on the density of attractive destinations at shorter distances.

Mathematical formulation:

Define :math:`\Omega(i,j)` to be the set of all locations :math:`k` such that :math:`D_{i,k} \le D_{i,j}`

:math:`M_{i,j} = k p_i^a \sum_j \left(\frac{p_j}{\sum_{k \in \Omega(i,j)} p_k}\right)^b`

This presents us with the choice of whether or not the origin population :math:`i` is included in :math:`\Omega` – i.e., does the same "gravity" that brings others to visit a community reduce the propensity of that community's members to travel to other communities?

The Stouffer model does not include impact from the local community:
:math:`\Omega(i,j) = \left(k: 0 < D_{i,k} \le D_{i,j}\right)`.

The Stouffer variant model does include the impact of the local community:
:math:`\Omega(i,j) = \left(k: 0 \le D_{i,k} \le D_{i,j}\right)`.

Rather than implementing twice, this implementation of the Stouffer model will include a parameter "include_home."

Radiation model
===============

The radiation model (Simini F, González MC, Maritan A, Barabási AL. A universal model for mobility and migration patterns. Nature. 2012;484(7392):96–100.) is a parameter-free model (up to an overall scaling constant for total migration flux), derived from arguments around job-related commuting but essentially capturing a situation in which outbound migration flux from origin to destination is enhanced by destination population and absorbed by the density of nearer destinations.

Mathematical formulation:
With :math:`\Omega` defined as above in the Stouffer model,

:math:`M_{i,j} = k \frac{p_i p_j}{\left(p_i + \sum_{k \in \Omega(i,j)} p_k\right)\left(p_i + p_j + \sum_{k \in \Omega(i,j)} p_k\right)}`

We again use the parameter "include_home" to determine whether or not location :math:`i` is to be included in :math:`\Omega(i,j)`.

Performance characteristics
===========================

All four migration models in :mod:`laser.core.migration` produce an :math:`N \times N`
network matrix from an :math:`N`-vector of populations and an :math:`N \times N`
distance matrix. Costs below are for a single call (network construction); they do
**not** include downstream sampling cost. Asymptotic complexity is the same for all
models — :math:`O(N^2)` — but the per-element constants differ.

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Model
     - Asymptotic cost
     - Inner loop
     - Notes
   * - :func:`laser.core.migration.gravity`
     - :math:`O(N^2)`
     - Fully vectorized NumPy
     - The cheapest model; one elementwise product and a fill of the diagonal.
   * - :func:`laser.core.migration.competing_destinations`
     - :math:`O(N^2)`
     - Fully vectorized NumPy
     - Vectorized in v1.0.2+ (previously an :math:`O(N^2)` Python double loop). One additional :math:`N \times N` adjustment matrix is built on top of the gravity output.
   * - :func:`laser.core.migration.stouffer`
     - :math:`O(N^2 \log N)`
     - Fully vectorized NumPy
     - Vectorized in v1.0.2+ (previously a per-source-node Python loop). Per-row sort dominates at large :math:`N`; equidistant ties handled correctly via grouped maxima.
   * - :func:`laser.core.migration.radiation`
     - :math:`O(N^2 \log N)`
     - Fully vectorized NumPy
     - Vectorized in v1.0.2+. Same per-row-sort cost as Stouffer; one additional element-wise division pair.
   * - :func:`laser.core.migration.distance`
     - :math:`O(N M)`
     - Fully vectorized broadcasting
     - Vectorized Haversine; takes two coordinate vectors of length :math:`N` and :math:`M` and returns the :math:`N \times M` great-circle distance matrix.

For up-to-date wall-clock numbers on representative inputs, run the benchmark suite
shipped under ``benchmarks/`` (see ``benchmarks/README.md`` for invocation and
``.github/workflows/benchmarks.yml`` for CI):

.. code-block:: bash

   pytest benchmarks/ --benchmark-only

The benchmark suite covers all four migration models and ``distance`` at multiple
scales (typically :math:`N = 100` and :math:`N = 1{,}000`) so that performance
regressions in any one of them are caught before merging.

.. note::
   Numerical behavior is pinned by the bit-equivalence regression tests in
   ``tests/test_migration.py`` (``TestMigrationVectorizationRegression``): every
   vectorized model is compared against a loop-based reference within float
   tolerance (``rtol`` :math:`\le 10^{-10}`). When extending these models, mirror
   the same test pattern.

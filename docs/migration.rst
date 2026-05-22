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

Because the inputs to spatial models (populations, distances) can vary over many orders of magnitude, we can run into situations where a a small number of nodes, often those closest to but distinct from large population centers, will end up with huge outflows. The below illustrates an easy way to implement a standard gravity/radiation/etc. model, but cap the total amount of migration/infectivity outflow from any single metapopulation.

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

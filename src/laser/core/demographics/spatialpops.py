"""Functions for distributing a total agent population heterogeneously across nodes.

Currently provides two strategies:

- [`distribute_population_skewed`][laser.core.demographics.spatialpops.distribute_population_skewed]:
  one large urban node plus an exponentially-tapered rural distribution with a hard
  per-node floor.
- [`distribute_population_tapered`][laser.core.demographics.spatialpops.distribute_population_tapered]:
  a smooth power-law / exponential taper across all nodes.

Both use the LASER-wide PRNG, so seeding via [`laser.core.random.seed`][laser.core.random.seed]
controls reproducibility.
"""

import numpy as np

from laser.core.random import prng


def distribute_population_skewed(tot_pop: int, num_nodes: int, frac_rural: float = 0.3, min_pop: int = 1000) -> np.ndarray:
    """
    Distribute a population across one urban node and several rural nodes.

    Node 0 is the single urban node and is assigned ``(1 - frac_rural) * tot_pop``
    agents. The remaining ``num_nodes - 1`` rural nodes collectively receive
    ``frac_rural * tot_pop`` agents, with their individual sizes drawn from an
    exponential distribution to create heterogeneity. Every node — urban and
    rural — is guaranteed at least ``min_pop`` agents.

    The function reserves ``min_pop`` for every rural node up front, draws the
    *surplus* (``frac_rural - (num_nodes - 1) * min_pop / tot_pop``) from an
    exponential distribution, normalizes the surplus so that the rural shares
    sum to ``frac_rural``, and then adds the reservation back. This construction
    guarantees the per-node floor without any post-hoc clipping, and keeps the
    sum exact (after a small rounding correction absorbed on the urban node).

    Parameters:
        tot_pop: Total population to distribute. Must be greater than 0.
        num_nodes: Total number of nodes (1 urban + ``num_nodes - 1`` rural).
            Must be greater than 0.
        frac_rural: Fraction of ``tot_pop`` assigned collectively to the
            rural nodes; the urban node receives ``1 - frac_rural``. Must be in
            ``[0, 1]``. Defaults to 0.3.
        min_pop: Minimum population guaranteed for every node, urban and
            rural alike. Must be non-negative. Defaults to 1000.

    Returns:
        np.ndarray: A 1-D ``uint32`` array of length ``num_nodes`` whose sum
        equals ``tot_pop``. Index 0 is the urban node; indices ``1..num_nodes-1``
        are the rural nodes.

    Raises:
        ValueError: If ``tot_pop <= 0``, ``num_nodes <= 0``, ``frac_rural`` is
            outside ``[0, 1]``, or ``min_pop < 0``.
        ValueError: If ``frac_rural == 0`` but ``num_nodes != 1``, or if
            ``frac_rural > 0`` but ``num_nodes < 2``. The function models one
            urban node plus zero or more rural nodes, so a positive rural
            budget requires at least one rural node and a zero rural budget
            requires zero rural nodes.
        ValueError: If the parameter combination is infeasible — i.e., the
            urban node cannot meet ``min_pop`` (``(1 - frac_rural) * tot_pop <
            min_pop``) or the per-rural-node average is below ``min_pop``
            (``frac_rural * tot_pop / (num_nodes - 1) < min_pop``). The tight
            case where the per-rural-node average equals ``min_pop`` is
            permitted; the exponential surplus is then zero and every rural
            node is pinned at exactly ``min_pop``. The error message
            identifies the binding constraint and which knobs to adjust.

    Notes:
        Random draws use the LASER-wide PRNG accessed via
        ``laser.core.random.prng()``. To make a result reproducible, call
        ``laser.core.random.seed(...)`` before invoking this function.

    **Example**:

        >>> from laser.core.random import seed
        >>> from laser.core.demographics.spatialpops import distribute_population_skewed
        >>> _ = seed(42)
        >>> pops = distribute_population_skewed(tot_pop=100_000, num_nodes=5, frac_rural=0.3, min_pop=1000)
        >>> int(pops.sum())
        100000
        >>> bool((pops >= 1000).all())
        True
        >>> int(pops[0])  # urban node has 1 - frac_rural of tot_pop
        70000
    """
    if tot_pop <= 0:
        raise ValueError("Total population must be greater than 0.")
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be greater than 0.")
    if not (0 <= frac_rural <= 1):
        raise ValueError("Fraction of rural population must be between 0 and 1.")
    if min_pop < 0:
        raise ValueError("Minimum node population must be non-negative.")

    # frac_rural == 0 ⟺ num_nodes == 1. A positive rural budget needs at least
    # one rural node; conversely, multiple nodes need a positive rural budget
    # (otherwise the rural nodes are empty by construction).
    if frac_rural == 0.0 and num_nodes != 1:
        raise ValueError(
            f"frac_rural=0 requires num_nodes=1 (no rural budget means no rural "
            f"nodes); got num_nodes={num_nodes}. Either set frac_rural > 0 or "
            f"set num_nodes=1."
        )
    if frac_rural > 0.0 and num_nodes < 2:
        raise ValueError(
            f"frac_rural={frac_rural} > 0 requires num_nodes >= 2 (at least one "
            f"rural node to receive the rural budget); got num_nodes={num_nodes}. "
            f"Either set frac_rural=0 or increase num_nodes."
        )

    num_rural = num_nodes - 1
    urban_pop = (1 - frac_rural) * tot_pop
    rural_total = frac_rural * tot_pop

    if urban_pop < min_pop:
        raise ValueError(
            f"Urban node would receive {urban_pop:.0f} agents, below "
            f"min_pop={min_pop}. Reduce frac_rural, reduce min_pop, or "
            f"increase tot_pop."
        )
    # Non-strict feasibility: when the rural budget exactly equals
    # num_rural * min_pop, every rural node is pinned at min_pop (zero
    # exponential surplus to spread). Only reject when the budget is smaller.
    if num_rural > 0 and rural_total < num_rural * min_pop:
        per_node_avg = rural_total / num_rural
        raise ValueError(
            f"Rural per-node average ({per_node_avg:.2f} agents) is below "
            f"min_pop={min_pop}: rural budget cannot accommodate {num_rural} "
            f"rural nodes at the floor (need "
            f"frac_rural * tot_pop / (num_nodes - 1) >= min_pop). "
            f"Increase tot_pop or frac_rural, or reduce num_nodes or min_pop."
        )

    if num_rural == 0:
        # Single-node case (frac_rural == 0): all population goes to the urban node.
        return np.array([tot_pop], dtype=np.uint32)

    # Reserve min_pop for every rural node; draw the surplus from an exponential.
    # max(0.0, ...) clamps FP drift in the subtraction so surplus_frac is never
    # spuriously negative when the tight-budget case is in play.
    min_frac = min_pop / tot_pop
    surplus_frac = max(0.0, frac_rural - num_rural * min_frac)

    rng = prng()
    weights = rng.exponential(size=num_rural)
    if surplus_frac > 0:
        weights *= surplus_frac / weights.sum()
    else:
        # Tight budget: every rural node lands at exactly min_pop.
        weights[:] = 0.0

    # Each rural fraction is min_frac plus a non-negative surplus weight; rural
    # fractions sum to frac_rural by construction.
    rural_fracs = min_frac + weights
    nsizes = np.insert(rural_fracs, 0, 1 - frac_rural)

    # Round to integer agent counts, then absorb the sub-unit rounding residual
    # on the urban node — it has the most slack above min_pop and is not part
    # of the random draw, so adjusting it leaves the rural distribution intact.
    npops = np.round(tot_pop * nsizes).astype(np.int64)
    npops[0] += tot_pop - npops.sum()

    return npops.astype(np.uint32)


def distribute_population_tapered(tot_pop, num_nodes):
    """
    Distribute a total population heterogeneously across a given number of nodes.

    The distribution follows a logarithmic-like decay pattern where the first node
    (Node 0) receives the largest share of the population, approximately half the
    total population. Subsequent nodes receive progressively smaller populations,
    ensuring that even the smallest node has a non-negligible share.

    The function ensures the sum of the distributed populations matches the
    `tot_pop` exactly by adjusting the largest node if rounding introduces discrepancies.

    Parameters:
        tot_pop (int): The total population to distribute. Must be a positive integer.
        num_nodes (int): The number of nodes to distribute the population across. Must be a positive integer.

    Returns:
        ndarray (numpy.ndarray): A 1D array of integers where each element represents the population assigned
            to a specific node. The length of the array is equal to `num_nodes`.

    Raises:
        ValueError: If `tot_pop` or `num_nodes` is not greater than 0.

    Notes:
        - The logarithmic-like distribution ensures that Node 0 has the highest population,
          and subsequent nodes receive progressively smaller proportions.
        - The function guarantees that the sum of the returned array equals `tot_pop`.

    Examples:

        Distribute a total population of 1000 across 5 nodes::

            >>> from laser.core.demographics.spatialpops import distribute_population_tapered
            >>> distribute_population_tapered(1000, 5)
            array([500, 250, 125, 75, 50])

        Distribute a total population of 1200 across 3 nodes::

            >>> distribute_population_tapered(1200, 3)
            array([600, 400, 200])

        Handling a small total population with more nodes::

            >>> distribute_population_tapered(10, 4)
            array([5, 3, 2, 0])

        Ensuring the distribution adds up to the total population::

            >>> pop = distribute_population_tapered(1000, 5)
            >>> pop.sum()
            1000
    """
    if num_nodes <= 0 or tot_pop <= 0:
        raise ValueError("Both tot_pop and num_nodes must be greater than 0.")

    # Generate a logarithmic-like declining distribution
    weights = np.logspace(0, -1, num=num_nodes, base=10)  # Declines logarithmically
    weights = weights / weights.sum()  # Normalize weights to sum to 1

    # Scale weights to the total population and round to integers
    population_distribution = np.round(weights * tot_pop).astype(int)

    # Ensure the sum matches the tot_pop by adjusting the largest node
    difference = tot_pop - population_distribution.sum()
    population_distribution[0] += difference  # Adjust Node 0 (largest) to make up the difference

    return population_distribution

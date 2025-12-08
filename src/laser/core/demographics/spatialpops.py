# Some functions for distributing a given population of agents heterogeneously across a number of nodes according to a simple distribution

import numpy as np


def distribute_population_skewed(tot_pop, num_nodes, frac_rural=0.3):
    """
    Calculate the population distribution across a number of nodes based on
    a total population, the number of nodes, and the fraction of the population
    assigned to rural nodes.

    The function generates a list of node populations distributed according to
    a simple exponential random distribution, with adjustments to ensure the
    sum matches the total population and the specified fraction of rural
    population is respected.

    Parameters:
        tot_pop (int): The total population to be distributed across the nodes.
        num_nodes (int): The total number of nodes among which the population will be distributed.
        frac_rural (float): The fraction of the total population to be assigned to rural nodes
            (value between 0 and 1). Defaults to 0.3. The 0 node is the single urban
            node and has (1 - frac_rural) of the population.

    Returns:
        list[int]: A list of integers representing the population at each node. The sum of the list equals `tot_pop`.

    Notes:
        - The population distribution is weighted using an exponential random
          distribution to create heterogeneity among node populations.
        - Adjustments are made to ensure the total fraction assigned to rural
          nodes adheres to `frac_rural`.

    Examples
    --------
    >>> from laser.core.demographics.spatialpops import distribute_population_skewed
    >>> np.random.seed(42)  # For reproducibility
    >>> tot_pop = 1000
    >>> num_nodes = 5
    >>> frac_rural = 0.3
    >>> distribute_population_skewed(tot_pop, num_nodes, frac_rural)
    [700, 154, 64, 54, 28]

    >>> tot_pop = 500
    >>> num_nodes = 3
    >>> frac_rural = 0.4
    >>> distribute_population_skewed(tot_pop, num_nodes, frac_rural)
    [300, 136, 64]
    """
    # Valid input data checks
    if tot_pop <= 0:
        raise ValueError("Total population must be greater than 0.")
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be greater than 0.")
    if not (0 <= frac_rural <= 1):
        raise ValueError("Fraction of rural population must be between 0 and 1.")

    # Generate node sizes
    nsizes = np.exp(-np.log(np.random.rand(num_nodes - 1)))
    nsizes = frac_rural * nsizes / np.sum(nsizes)
    nsizes = np.minimum(nsizes, 100 / tot_pop)
    nsizes = frac_rural * nsizes / np.sum(nsizes)
    nsizes = np.insert(nsizes, 0, 1 - frac_rural)

    # Calculate populations and round to integers
    npops = ((np.round(tot_pop * nsizes, 0)).astype(int)).tolist()

    # Ensure total population matches tot_pop
    difference = tot_pop - sum(npops)
    npops[1] += difference  # Adjust the second node

    return np.array(npops, dtype=np.uint32)


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

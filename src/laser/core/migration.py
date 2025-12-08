r"""
This module provides various functions to calculate migration networks based on different models,
including the gravity model, competing destinations model, Stouffer's model, and the radiation model.

Additionally, it includes a utility function to calculate the great-circle distance between two points
on the Earth's surface using the Haversine formula.

Functions:
    gravity(pops: np.ndarray, distances: np.ndarray, k: float, a: float, b: float, c: float, max_frac: Union[float, None]=None, kwargs) -> np.ndarray:
        Compute a gravity-model migration network based on origin and destination populations and pairwise distances.

    row_normalizer(network: np.ndarray, max_rowsum: float) -> np.ndarray:
        Normalize the rows of a given network matrix such that no row sum exceeds a specified maximum value.

    competing_destinations(pops: np.ndarray, distances: np.ndarray, b: float, c: float, delta: float, params) -> np.ndarray:
        Compute a migration network using the competing-destinations model, incorporating the influence of alternative destination nodes.

    stouffer(pops: np.ndarray, distances: np.ndarray, k: float, a: float, b: float, include_home: bool, params) -> np.ndarray:
        Compute a migration network using a modified Stouffer's model.

    radiation(pops: np.ndarray, distances: np.ndarray, k: float, include_home: bool, params) -> np.ndarray:
        Compute a migration network using the radiation model, which models flows based on intervening population rather than physical distance.

    distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        Calculate the great-circle distance between two points on the Earth's surface
        using the Haversine formula.
"""
from numbers import Number

import numpy as np


def gravity(pops: np.ndarray, distances: np.ndarray, k: float, a: float, b: float, c: float, **kwargs):
    r"""
    Calculate a gravity model network.

    This function computes a gravity model network based on the provided populations
    and distances. The gravity model estimates migration or interaction flows between
    populations using a mathematical formula that incorporates scaling, population sizes,
    and distances.

    Mathematical formula::
        $$
        network_{i,j} = k \cdot \frac{p_i^a \cdot p_j^b}{distance_{i,j}^c}
        $$

    As implemented in NumPy::

        network = k * (pops[:, np.newaxis] ** a) * (pops ** b) * (distances ** (-1 * c))

    Parameters:
        pops (numpy.ndarray): 1D array of population sizes for each node.
        distances (numpy.ndarray): 2D array of distances between nodes. Must be symmetric, with self-distances (diagonal) handled.
        k (float): Scaling constant to adjust the overall magnitude of interaction flows.
        a (float): Exponent for the population size of the origin node.
        b (float): Exponent for the population size of the destination node.
        c (float): Exponent for the distance between nodes, controlling how distance impacts flows.
        kwargs (dict): Additional keyword arguments (not used in the current implementation).

    Returns:
        network (numpy.ndarray): A 2D matrix representing the interaction network, where each element `network[i, j]` corresponds to the flow from node `i` to node `j`.

    Example usage::

        import numpy as np
        from gravity_model import gravity

        populations = np.array([1000, 500, 200])
        distances = np.array([
            [0, 2, 3],
            [2, 0, 1],
            [3, 1, 0]
        ])

        k = 0.5
        a = 1.0
        b = 1.0
        c = 2.0

        migration_network = gravity(populations, distances, k, a, b, c)

        print("Migration Network:")
        print(migration_network)

    Notes:
        - The diagonal of the distances matrix is set to 1 internally to avoid division by zero.
        - The diagonal of the output network is set to 0 to represent no self-loops.
        - Ensure the distances matrix is symmetric and non-negative.
    """
    # Ensure pops and distances are valid
    _sanity_checks(pops, distances, a=a, b=b, c=c, k=k)

    # Promote pops to a robust datatype (e.g., int32 is likely to overflow)
    pops = pops.astype(np.float64)

    # Prevent division by zero by setting diagonal to 1
    distances1 = distances.copy().astype(np.float64)
    np.fill_diagonal(distances1, 1)

    # Compute the gravity model network
    network = k * (pops[:, np.newaxis] ** a) * (pops**b) * (distances1 ** (-1 * c))

    # Set the diagonal of the network to 0
    np.fill_diagonal(network, 0)

    return network


def row_normalizer(network, max_rowsum):
    """
    Normalizes the rows of a given network matrix such that no row sum exceeds a specified maximum value.

    Parameters:
        network (numpy.ndarray): A 2D array representing the network matrix.
        max_rowsum (float): The maximum allowable sum for any row in the network matrix.

    Returns:
        network (numpy.ndarray): The normalized network matrix where no row sum exceeds the specified maximum value.
    """

    # Sanity checks
    _is_instance(network, np.ndarray, f"network must be a NumPy array ({type(network)=})")
    _is_dtype(network, np.number, f"network must be a numeric array ({network.dtype=})")
    _has_dimensions(network, 2, f"network must be a 2D array ({network.shape=})")
    _has_shape(network, (network.shape[0], network.shape[0]), f"network must be a square matrix ({network.shape=})")
    _has_values(network >= 0, "network must contain only non-negative values")

    _is_instance(max_rowsum, Number, f"max_rowsum must be a numeric value ({type(max_rowsum)=})")
    _has_values(0 <= max_rowsum <= 1, "max_rowsum must be in [0, 1]")

    # Promote network to floating point
    # If the incoming network is an integer type and max_rowsum is < 1.0, the result is all zeros.
    network = network.copy().astype(np.float32)

    rowsums = network.sum(axis=1)
    rows_to_renorm = rowsums > max_rowsum
    network[rows_to_renorm] = network[rows_to_renorm] * max_rowsum / rowsums[rows_to_renorm, np.newaxis]

    return network


def competing_destinations(pops, distances, k, a, b, c, delta, **params):
    r"""
    Calculate the competing destinations model for a given set of populations and distances. (Fotheringham AS. Spatial flows and spatial patterns. Environment and planning A. 1984;16(4):529-543)

    This function computes a network matrix based on the gravity model and then adjusts it
    using the competing destinations model. The adjustment is done by considering the
    interference from other destinations.

    Mathematical formula:
        Element-by-element:
            $$
            network_{i,j} = k \times p_i^a \times p_j^b / distance_{i,j}^c \times \sum_k {(p_k^b / distance_{j,k}^c \text {\small for k not in [i,j]})^{delta} }
            $$

        As-implemented numpy math:

        - Compute all terms up to the sum_k using the gravity model
        - Construct the matrix inside the sum: ``p**b * distances**(1-c)``
        - Sum on the second axis (k), and subtract off the diagonal (j=k terms):
            ``row_sums = np.sum(competition_matrix, axis=1) - np.diag(competition_matrix)``
        - Now element-by-element, subtract k=i terms off the sum, exponentiate, and multiply the original network term:
            ``network[i][j] = network[i][j] * (row_sums[i] - competition_matrix[i][j]) ** delta``

    Parameters:
        pops (numpy.ndarray): Array of populations.
        distances (numpy.ndarray): Array of distances between locations.
        k (float): Scaling constant.
        a (float): Exponent for the population of the origin.
        b (float): Exponent parameter for populations in the gravity model.
        c (float): Exponent parameter for distances in the gravity model.
        delta (float): Exponent parameter for the competing destinations adjustment.
        params (dict): Additional parameters to be passed to the gravity model.

    Returns:
        network (numpy.ndarray): Adjusted network matrix based on the competing destinations model.
    """

    # Sanity checks
    _sanity_checks(pops, distances, a=a, k=k, b=b, c=c, delta=delta)

    # Promote pops to a robust datatype (e.g., int32 is likely to overflow)
    pops = pops.astype(np.float64)

    network = gravity(pops, distances, k=k, a=a, b=b, c=c, **params)
    # Construct the p_j^b / d_jk^c matrix, inside the sum
    distances1 = distances.copy().astype(np.float64)
    np.fill_diagonal(distances1, 1)  # Prevent division by zero in `distances ** (-1 * c)`
    competition_matrix = pops**b * distances1 ** (-1 * c)

    # Sum over all k, and remove the terms j=k
    # Don't subtract off the diagonal - this could be a NaN if distance to self is zero
    np.fill_diagonal(competition_matrix, 0)
    mysums = np.sum(competition_matrix, axis=1)
    # Rather than computing that interior sum for each element, just compute the row sums,
    # and then subtract the k=i term.
    for i in range(len(pops)):
        for j in range(len(pops)):
            if j != i:
                network[i][j] = network[i][j] * (mysums[j] - competition_matrix[j][i]) ** delta

    np.fill_diagonal(network, 0)
    return network


def sum_populations_as_close_or_closer(sorted_pops, sorted_distance_row):
    # Separating this operation out because it is common between a couple of migration models, and there is a little trickiness in appropriately
    # handling cases where there are multiple destinations equidistant from the source node.
    _is_instance(sorted_pops, np.ndarray, f"sorted_pops must be a NumPy array ({type(sorted_pops)=})")
    _is_dtype(sorted_pops, np.number, f"sorted_pops must be a numeric array ({sorted_pops.dtype=})")
    _has_values(sorted_pops >= 0, "sorted_pops must contain only non-negative values")
    _is_instance(sorted_distance_row, np.ndarray, f"sorted_distance_row must be a NumPy array ({type(sorted_distance_row)=})")
    _is_dtype(sorted_distance_row, np.number, f"sorted_distance_row must be a numeric array ({sorted_distance_row.dtype=})")
    _has_values(sorted_distance_row >= 0, "sorted_distance_row must contain only non-negative values")
    _has_shape(
        sorted_pops,
        sorted_distance_row.shape,
        f"sorted_pops & sorted_distance_row must have same shape ({sorted_pops.shape=}, {sorted_distance_row.shape=})",
    )
    _has_values(np.diff(sorted_distance_row) >= 0, "sorted_distance_row must be sorted in ascending order")

    # if all distances from the source node are unique, this is easy
    cumulative_sorted_pops = np.cumsum(sorted_pops)

    if len(np.unique(sorted_distance_row)) < len(sorted_distance_row):
        # However, if there are non-unique distances, then we have to account for each equidistant node in the sum,
        # as these models tend to sum over "all nodes k such that d_ik <= d_ij".
        # Since the vector is sorted, this just means to find the first and last occurences of occurence of each unique distance,
        # and replace the cumulative sum of populations for all of those terms with the last one.
        _, start_indices = np.unique(sorted_distance_row, return_index=True)
        _, end_indices = np.unique(sorted_distance_row[::-1], return_index=True)
        end_indices = len(sorted_distance_row) - end_indices - 1

        for idx in range(len(start_indices)):
            cumulative_sorted_pops[start_indices[idx] : end_indices[idx]] = cumulative_sorted_pops[end_indices[idx]]

    return cumulative_sorted_pops


def stouffer(pops, distances, k, a, b, include_home, **params):
    r"""
    Computes a migration network using a modified Stouffer's model.

    (Stouffer SA. Intervening opportunities: a theory relating mobility and distance. American sociological review. 1940;5(6):845-867)

    Mathematical formula:
        Element-by-element:
            $$
            network_{i,j} = k \times p_i \times p_j / ( (p_i + \sum_k {p_k}) (p_i + p_j + \sum_k {p_k}) )
            $$
            the parameter ``include_home`` determines whether $p_i$ is included or excluded from the sum

        As-implemented numpy math:

        - Sort each row of the distance matrix (we'll use \' below to indicate distance-sorted vectors)
        - Loop over "source nodes" i:

            - Cumulative sum the sorted populations, ensuring appropriate handling when there are multiple destinations equidistant from the source
            - Subtract the source node population if ``include_home`` is ``False``
            - Construct the row of the network matrix as $k \times p_i^a \times (p_{j'} / \sum_{k'} {p_{k'}})^b$

        - Unsort the rows of the network

    Parameters:

        pops (numpy.ndarray): An array of population sizes.
        distances (numpy.ndarray): A 2D array where `distances[i][j]` is the distance from location i to location j.
        k (float): A scaling factor for the migration rates.
        a (float): Exponent applied to the population size of the origin.
        b (float): Exponent applied to the ratio of destination population to the sum of all populations at equal or lesser distances.
        include_home (bool): If True, includes the home population in the cumulative sum; otherwise, excludes it.
        **params: Additional parameters (not used in the current implementation).

    Returns:
        network (numpy.ndarray): A 2D array representing the migration network, where `network[i][j]` is the migration rate from location i to location j.
    """

    # Sanity checks
    _sanity_checks(pops, distances, a=a, b=b, k=k, include_home=include_home)

    # Promote pops and distances to a robust datatype (e.g., int32 is likely to overflow)
    pops = pops.astype(np.float64)
    distances = distances.astype(np.float64)

    # We will just use the "truthiness" of include_home (could be boolean, could be 0/1)

    network = np.zeros_like(distances)
    sort_indices = np.argsort(distances, axis=1, kind="stable")
    unsort_indices = np.argsort(sort_indices, axis=1)

    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]
        cumulative_sorted_pops = sum_populations_as_close_or_closer(sorted_pops, distances[i][sort_indices[i]])

        if not include_home:
            cumulative_sorted_pops = cumulative_sorted_pops - sorted_pops[0]

        network[i, 1:] = k * pops[i] ** a * (sorted_pops[1:] / cumulative_sorted_pops[1:]) ** b
    network = np.take_along_axis(network, unsort_indices, axis=1)

    np.fill_diagonal(network, 0)
    return network


def radiation(pops, distances, k, include_home, **params):
    r"""
    Calculate the migration network using the radiation model.

    (Simini F, González MC, Maritan A, Barabási AL. A universal model for mobility and migration patterns. Nature. 2012;484(7392):96-100.)

    Mathematical formula:
        Element-by-element:
            $$
            network_{i,j} = k \times p_i^a \times (p_j / \sum_k {p_k} )^b,
            $$
            where the sum proceeds over all $k$ such that $distances_{i,k} \leq distances_{i,j}$
            the parameter ``include_home`` determines whether $p_i$ is included or excluded from the sum.

        As-implemented numpy math:

        - Sort each row of the distance matrix (we'll use \' below to indicate distance-sorted vectors)
        - Loop over "source nodes" i

            - Cumulative sum the sorted populations, ensuring appropriate handling when there are multiple destinations equidistant from the source
            - Subtract the source node population if ``include_home`` is ``False``
            - Construct the row of the network matrix as
                $$
                k \times p_i \times p_{j'} / (p_i + \sum_{k'} {p_{k'}}) / (p_i + p_{j'} + \sum_{k'} {p_{k'}})
                $$

        - Unsort the rows of the network

    Parameters:
        pops (numpy.ndarray): Array of population sizes for each node.
        distances (numpy.ndarray): 2D array of distances between nodes.
        k (float): Scaling factor for the migration rates.
        include_home (bool): Whether to include the home population in the calculations.
        params (dict): Additional parameters (currently not used).

    Returns:
        network (numpy.ndarray): 2D array representing the migration network.
    """

    # Sanity checks
    _sanity_checks(pops, distances, k=k, include_home=include_home)

    # Promote pops and distances to a robust datatype (e.g., int32 is likely to overflow)
    pops = pops.astype(np.float64)
    distances = distances.astype(np.float64)

    # We will just use the "truthiness" of include_home (could be boolean, could be 0/1)

    network = np.zeros_like(distances)
    sort_indices = np.argsort(distances, axis=1, kind="stable")
    unsort_indices = np.argsort(sort_indices, axis=1)

    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]
        cumulative_sorted_pops = sum_populations_as_close_or_closer(sorted_pops, distances[i][sort_indices[i]])

        if not include_home:
            cumulative_sorted_pops = cumulative_sorted_pops - sorted_pops[0]

        network[i] = k * pops[i] * sorted_pops / (pops[i] + cumulative_sorted_pops) / (pops[i] + sorted_pops + cumulative_sorted_pops)

    network = np.take_along_axis(network, unsort_indices, axis=1)
    np.fill_diagonal(network, 0)
    return network


def distance(lat1, lon1, lat2=None, lon2=None):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    This function uses the Haversine formula to compute the distance between two points
    specified by their latitude and longitude in decimal degrees.

    - If `lat2` and `lon2` are not provided, they default to `lat1` and `lon1`, respectively. This supports the default case of calculating the NxN matrix of distances between all pairs of points in (`lat1`, ` lon1`).
    - If all arguments are scalars, will return a single scalar distance, (`lat1`, `lon1`) to (`lat2`, `lon2`).
    - If `lat2`, `lon2` are vectors, will return a vector of distances, (`lat1`, `lon1`) to each lat/lon in `lat2`, `lon2`.
    - If `lat1`, `lon1` and `lat2`, `lon2` are vectors, will return a matrix with shape (N, M) of distances where N is the length of `lat1`/`lon1` and M is the length of `lat2`/`lon2`.

    Parameters:
        lat1 (float): Latitude of the first point(s) in decimal degrees [-90, 90].
        lon1 (float): Longitude of the first point(s) in decimal degrees [-180, 180].
        lat2 (float): Latitude of the second point(s) in decimal degrees [-90, 90].
        lon2 (float): Longitude of the second point(s) in decimal degrees [-180, 180].

    Returns:
        distance (float): The distance between the points in kilometers.
    """

    # Sanity checks, part 1
    _is_instance(lat1, (Number, np.ndarray), f"lat1 must be a numeric value or NumPy array ({type(lat1)=})")
    _is_instance(lon1, (Number, np.ndarray), f"lon1 must be a numeric value or NumPy array ({type(lon1)=})")
    _has_values((-90 <= lat1) & (lat1 <= 90), "lat1 must be in the range [-90, 90]")
    _has_values((-180 <= lon1) & (lon1 <= 180), "lon1 must be in the range [-180, 180]")

    if (lat2 is None) and (lon2 is None):
        lat2 = lat1
        lon2 = lon1

    if (lat2 is None) or (lon2 is None):
        raise ValueError("Either both or neither of lat2 and lon2 must be provided.")

    # Sanity checks, part 2
    _is_instance(lat2, (Number, np.ndarray), f"lat2 must be a numeric value or NumPy array ({type(lat2)=})")
    _is_instance(lon2, (Number, np.ndarray), f"lon2 must be a numeric value or NumPy array ({type(lon2)=})")
    _has_values((-90 <= lat2) & (lat2 <= 90), "lat2 must be in the range [-90, 90]")
    _has_values((-180 <= lon2) & (lon2 <= 180), "lon2 must be in the range [-180, 180]")

    lat1 = np.array(lat1).flatten()
    lon1 = np.array(lon1).flatten()
    lat2 = np.array(lat2).flatten()
    lon2 = np.array(lon2).flatten()

    _has_shape(lon1, lat1.shape, f"lat1 and lon1 must have the same shape ({lat1.shape=}, {lon1.shape=})")
    _has_shape(lon2, lat2.shape, f"lat2 and lon2 must have the same shape ({lat2.shape=}, {lon2.shape=})")

    # convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    d = np.zeros((lat1.size, lat2.size))
    for index in range(lat1.size):
        # haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)
        dlat = lat2 - lat1[index]
        dlon = lon2 - lon1[index]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1[index]) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        d[index, :] = a
    d = 2 * np.arcsin(np.sqrt(d))
    RE = 6371.0  # Earth radius in km
    d *= RE

    if d.size == 1:
        return d[0, 0]  # return a scalar
    elif np.any(np.array(d.shape) == 1):
        return d.reshape((d.size,))  # return a vector (1-D)

    return d  # return NxM matrix (len(lat1/lon1) x len(lat2/lon2))


# Sanity checks


def _sanity_checks(pops, distances, **params):
    _is_instance(pops, np.ndarray, f"pops must be a NumPy array ({type(pops)=})")
    _has_dimensions(pops, 1, f"pops must be a 1D array ({pops.shape=})")
    _is_dtype(pops, np.number, f"pops must be a numeric array ({pops.dtype=})")
    _has_values(pops >= 0, "pops must contain only non-negative values")

    _is_instance(distances, np.ndarray, f"distances must be a NumPy array ({type(distances)=})")
    _has_dimensions(distances, 2, f"distances must be a 2D array ({distances.shape=})")
    _has_shape(
        distances,
        (pops.shape[0], pops.shape[0]),
        f"distances must be a square matrix with length equal to the length of pops ({distances.shape=}, {pops.shape=})",
    )
    _is_dtype(distances, np.number, f"distances must be a numeric array ({distances.dtype=})")
    _has_values(distances >= 0, "distances must contain only non-negative values")
    _has_values(distances == distances.T, "distances must be a symmetric matrix")

    if "a" in params:
        a = params.get("a", None)
        _is_instance(a, Number, f"a must be a numeric value ({type(a)=})")
        _has_values(a >= 0, f"a must be a non-negative value ({a=})")

    if "b" in params:
        b = params.get("b", None)
        _is_instance(b, Number, f"b must be a numeric value ({type(b)=})")
        _has_values(b >= 0, f"b must be a non-negative value ({b=})")

    if "c" in params:
        c = params.get("c", None)
        _is_instance(c, Number, f"c must be a numeric value ({type(c)=})")
        _has_values(c >= 0, f"c must be a non-negative value ({c=})")

    if "delta" in params:
        delta = params.get("delta", None)
        _is_instance(delta, Number, f"delta must be a numeric value ({type(delta)=})")

    if "k" in params:
        k = params.get("k", None)
        _is_instance(k, Number, f"k must be a numeric value ({type(k)=})")
        _has_values(k >= 0, f"k must be a non-negative value ({k=})")

    if "include_home" in params:
        include_home = params.get("include_home", None)
        _is_instance(include_home, (int, bool), f"include_home must be boolean or integer type ({type(include_home)=})")


def _is_instance(obj, types, message):
    if not isinstance(obj, types):
        raise TypeError(message)

    return


def _has_dimensions(obj, dimensions, message):
    if not len(obj.shape) == dimensions:
        raise TypeError(message)

    return


def _is_dtype(obj, dtype, message):
    if not np.issubdtype(obj.dtype, dtype):
        raise TypeError(message)

    return


def _has_values(check, message):
    if not np.all(check):
        raise ValueError(message)

    return


def _has_shape(obj, shape, message):
    if not obj.shape == shape:
        raise TypeError(message)

    return

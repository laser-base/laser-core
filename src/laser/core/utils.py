"""
This module provides utility functions for the laser-measles project.

Functions:
    calc_capacity(birthrates: np.ndarray, initial_pop: np.ndarray, safety_factor: float = 1.0) -> np.ndarray:
        Calculate the population capacity after a given number of ticks based on a constant birth rate.

    grid(shape: Tuple[int, int], fill_value: float = 0.0) -> np.ndarray:
        Create a 2D grid (numpy array) of the specified shape, filled with the given value.

"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def calc_capacity(birthrates: np.ndarray, initial_pop: np.ndarray, safety_factor: float = 1.0) -> np.ndarray:
    """
    Estimate the required capacity (number of agents) to model a population given birthrates over time.

    Args:

        birthrates (np.ndarray): 2D array of shape (nsteps, nnodes) representing birthrates (CBR) per 1,000 individuals per year.
        initial_pop (np.ndarray): 1D array of length nnodes representing the initial population at each node.
        safety_factor (float): Safety factor to account for variability in population growth. Default is 1.0.

    Returns:

        np.ndarray: 1D array of length nnodes representing the estimated required capacity (number of agents) at each node.
    """
    # Validate birthrates shape against initial_pop shape
    _, nnodes = birthrates.shape
    assert len(initial_pop) == nnodes, f"Number of nodes in birthrates ({nnodes}) and initial_pop length ({len(initial_pop)}) must match"

    # Validate birthrates values, must be >= 0 and <= 100
    assert np.all(birthrates >= 0.0), "All birthrate values must be non-negative"
    assert np.all(birthrates <= 100.0), "All birthrate values must be less than or equal to 100"

    # Validate safety_factor
    assert 0 <= safety_factor <= 6, f"safety_factor must be between 0 and 6, got {safety_factor}"

    # Convert CBR to daily growth rate
    # CBR = births per 1,000 individuals per year
    # CBR / 1000 = births per individual per year
    # Growth = (1 + CBR / 1000) per individual per year
    # Daily growth = (1 + CBR / 1000) ^ (1/365)
    # Daily growth rate = (1 + CBR / 1000) ^ (1/365) - 1
    # Note, sticking with "lamda" here a) for consistency with modern Greek and Unicode, b) to avoid confusion with Python keyword "lambda"
    lamda = (1.0 + birthrates / 1000) ** (1.0 / 365) - 1.0

    # Geometric Brownian motion approximation for population growth (https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Properties)
    # E(P_t) = P_0 * exp(mu * t)
    # where mu is the daily growth rate and t is the number of time steps (days)

    # Since we may have time-varying rates, add up daily growth rates over all time steps
    # Consider alternative: np.prod(1 + lamda, axis=0)
    # For 0 <= CBR <= 40, difference is negligible (< 1:1e6)
    exp_mu_t = np.exp(lamda.sum(axis=0))

    safety_multiplier = 1 + safety_factor * (np.sqrt(exp_mu_t) - 1)
    estimates = np.round(initial_pop * safety_multiplier * exp_mu_t).astype(np.int32)

    return estimates


def grid(M=5, N=5, node_size_degs=0.08983, population_fn=None, origin_x=0, origin_y=0):
    """
    Create an MxN grid of cells anchored at (lat, long) with populations and geometries.

    Args:
        M (int): Number of rows (north-south).
        N (int): Number of columns (east-west).
        node_size_degs (float): Size of each cell in decimal degrees (default 0.08983 ≈ 10km at the equator).
        population_fn (callable): Function(row, col) returning population for a cell. Default is uniform random between 1,000 and 100,000.
        origin_x (float): longitude of the origin in decimal degrees (bottom-left corner) -180 <= origin_x < 180.
        origin_y (float): latitude of the origin in decimal degrees (bottom-left corner) -90 <= origin_y < 90.

    Returns:
        scenario (GeoDataFrame): Columns are nodeid, population, geometry.
    """

    if M < 1:
        raise ValueError("M must be >= 1")
    if N < 1:
        raise ValueError("N must be >= 1")
    if node_size_degs <= 0:
        raise ValueError("node_size_degs must be > 0")
    if node_size_degs > 1.0:
        raise ValueError("node_size_degs must be <= 1.0")
    if not (-180 <= origin_x < 180):
        raise ValueError("origin_x must be -180 <= origin_x < 180")
    if not (-90 <= origin_y < 90):
        raise ValueError("origin_y must be -90 <= origin_y < 90")

    if population_fn is None:

        def population_fn(row: int, col: int) -> int:
            return int(np.random.uniform(1_000, 100_000))

    cells = []
    nodeid = 0
    for row in range(M):
        for col in range(N):
            # TODO - use latitude sensitive conversion of km to degrees
            x0 = origin_x + col * node_size_degs
            y0 = origin_y + row * node_size_degs
            x1 = x0 + node_size_degs
            y1 = y0 + node_size_degs
            poly = Polygon(
                [
                    (x0, y0),  # SW
                    (x1, y0),  # SE
                    (x1, y1),  # NE
                    (x0, y1),  # NW
                    (x0, y0),  # Close polygon in SW
                ]
            )
            population = int(population_fn(row, col))
            if population < 0:
                raise ValueError(f"population_fn returned negative population {population} for row {row}, col {col}")
            cells.append({"nodeid": nodeid, "population": population, "geometry": poly})
            nodeid += 1

    gdf = gpd.GeoDataFrame(cells, columns=["nodeid", "population", "geometry"], crs="EPSG:4326")

    return gdf

"""A class for generating samples from a distribution using the Vose alias method."""

import re
from pathlib import Path

import numpy as np

from laser.core.random import prng


class AliasedDistribution:
    """A class to generate samples from a distribution using the Vose alias method."""

    def __init__(self, counts):
        """Build the alias and probability tables for the Vose alias method.

        Args:
            counts (array-like of int): Bin counts (or relative weights) used to construct
                the discrete distribution. Must contain at least one positive value.

        Raises:
            ValueError: If `counts` has zero length or sums to zero (no normalization possible).

        **Example**:

            dist = AliasedDistribution([10, 20, 30, 40])
            sample = dist.sample(count=5)
        """
        # TODO, consider int64 or uint64 if using global population
        alias = np.full(len(counts), -1, dtype=np.int32)
        probs = np.array(counts, dtype=np.int32)
        total = probs.sum()
        # Scale by len(probs) so we can test bin probability against the average without division:
        # we want p[i] < p.sum() / len(p); rearranged to (p[i] * len(p)) < p.sum().
        probs *= len(probs)

        small = [i for i, value in enumerate(probs) if value < total]
        large = [i for i, value in enumerate(probs) if value > total]
        while small:
            ismall = small.pop()
            ilarge = large.pop()
            alias[ismall] = ilarge
            probs[ilarge] -= total - probs[ismall]
            if probs[ilarge] < total:
                small.append(ilarge)
            elif probs[ilarge] > total:
                large.append(ilarge)

        self._alias = alias
        self._probs = probs
        self._total = total

        return

    @property
    def alias(self) -> np.ndarray:
        """np.ndarray: The alias index lookup table used by the Vose alias method.

        Each entry `alias[i]` is the fallback bin chosen when a uniform draw against
        `probs[i]` indicates the current bin `i` should be substituted.
        """
        return self._alias

    @property
    def probs(self) -> np.ndarray:
        """np.ndarray: The per-bin acceptance thresholds (scaled by `len(probs)`).

        At sampling time, a bin `i` is accepted directly when a uniform draw in
        `[0, total)` is less than `probs[i]`; otherwise the alias bin is substituted.
        """
        return self._probs

    @property
    def total(self) -> int:
        """int: Sum of the original input counts, used as the uniform-draw upper bound."""
        return self._total

    def sample(self, count=1, dtype=np.int32) -> int:
        """
        Generate samples from the distribution.

        Parameters:
            count (int): The number of samples to generate. Default is 1.

        Returns:
            int or numpy.ndarray: A single integer if count is 1, otherwise an array of integers representing the generated samples.
        """

        rng = prng()

        if count == 1:
            i = rng.integers(low=0, high=len(self._alias), dtype=dtype)
            d = rng.integers(low=0, high=self._total, dtype=dtype)

            i = i if d < self._probs[i] else self._alias[i]
        else:
            i = rng.integers(low=0, high=len(self._alias), size=count, dtype=dtype)
            d = rng.integers(low=0, high=self._total, size=count, dtype=dtype)
            a = d >= self._probs[i]
            i[a] = self._alias[i[a]]

        return i


def load_pyramid_csv(file: Path, verbose=False) -> np.ndarray:
    """
    Load a CSV file with population pyramid data and return it as a NumPy array.

    The CSV file is expected to have the following schema:

    - The first line is a header: "Age,M,F"
    - Subsequent lines contain age ranges and population counts for males and females:

    ```
    "low-high,#males,#females"
    ...
    "max+,#males,#females"
    ```
    Where low, high, males, females, and max are integer values >= 0.

    The function processes the CSV file to create a NumPy array with the following columns:

    - Start age of the range
    - End age of the range
    - Number of males
    - Number of females

    Parameters:
        file (Path): The path to the CSV file.
        verbose (bool): If True, prints the file reading status. Default is False.

    Returns:
        np.ndarray: A NumPy array with the processed population pyramid data.
    """

    if verbose:
        print(f"Reading population pyramid data from '{file}' ...")
    file = Path(file)
    with file.open("r") as f:
        lines = [line.strip() for line in f.readlines()]  # Use strip to remove newline characters

    # Validate incoming text
    if not lines[0] == "Age,M,F":
        raise ValueError("Header line is not 'Age,M,F'.")
    if not all(re.match(r"\d+-\d+,\d+,\d+", line) for line in lines[1:-1]):
        raise ValueError("Data lines are not in the expected format 'low-high,males,females'.")
    if not re.match(r"\d+\+,\d+,\d+", lines[-1]):
        raise ValueError("Last data line is not in the expected format 'max+,males,females'.")

    text = lines[1:]  # Skip the first line
    text = [line.split(",") for line in text]  # Split each line by comma
    text = [line[0].split("-") + line[1:] for line in text]  # Split the first element by hyphen
    text[-1][0] = text[-1][0].replace("+", "")  # Remove the plus sign from the last element
    data = [list(map(int, line)) for line in text]  # Convert all elements to integers
    data[-1] = [
        data[-1][0],
        data[-1][0],
        *data[-1][1:],
    ]  # Make the last element a single year bucket

    datanp = np.array(data, dtype=np.int32)

    # Validity checks
    # Note, negative values will not pass the regex check above.
    if not np.all(datanp[:-1, 0] < datanp[1:, 0]):
        raise ValueError("Starting ages are not in ascending order.")
    if not np.all(datanp[:-1, 1] < datanp[1:, 1]):
        raise ValueError("Ending ages are not in ascending order.")

    return datanp

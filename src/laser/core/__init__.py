"""LASER (Light Agent Spatial modeling for ERadication) core package.

Top-level entry points:

- [`LaserFrame`][laser.core.LaserFrame] — dynamically allocated agent / patch storage
- [`PropertySet`][laser.core.PropertySet] — composable parameter sets
- [`SortedQueue`][laser.core.SortedQueue] — fixed-capacity priority queue

Submodules surfaced for discoverability:

- [`distributions`][laser.core.distributions] — Numba-accelerated samplers
- [`migration`][laser.core.migration] — gravity / radiation / Stouffer / competing-destinations
- [`random`][laser.core.random] — process-wide PRNG with seedable, reproducible state
- [`demographics`][laser.core.demographics] — population pyramid and Kaplan-Meier utilities
- [`utils`][laser.core.utils] — capacity estimation and spatial grid helpers
"""

from importlib.metadata import version

from .laserframe import LaserFrame
from .propertyset import PropertySet
from .sortedqueue import SortedQueue

__version__ = version("laser.core")

__all__ = [
    "LaserFrame",
    "PropertySet",
    "SortedQueue",
    "__version__",
]

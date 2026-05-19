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

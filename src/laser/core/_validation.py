"""Internal input-validation helpers shared across the package.

These helpers raise `TypeError` or `ValueError` with a caller-supplied message when the
checked condition fails, so call sites can stay declarative (one line per precondition)
without repeating the `if not ...: raise` boilerplate.

These are package-private and not part of the public API.
"""

import numpy as np


def _is_instance(obj, types, message):
    """Raise `TypeError(message)` unless `isinstance(obj, types)`."""
    if not isinstance(obj, types):
        raise TypeError(message)
    return


def _has_dimensions(obj, dimensions, message):
    """Raise `TypeError(message)` unless `obj.shape` has the expected number of dimensions."""
    if not len(obj.shape) == dimensions:
        raise TypeError(message)
    return


def _is_dtype(obj, dtype, message):
    """Raise `TypeError(message)` unless `obj.dtype` is a subtype of `dtype`."""
    if not np.issubdtype(obj.dtype, dtype):
        raise TypeError(message)
    return


def _has_values(check, message):
    """Raise `ValueError(message)` unless every element of the boolean array `check` is truthy."""
    if not np.all(check):
        raise ValueError(message)
    return


def _has_shape(obj, shape, message):
    """Raise `TypeError(message)` unless `obj.shape == shape`."""
    if not obj.shape == shape:
        raise TypeError(message)
    return

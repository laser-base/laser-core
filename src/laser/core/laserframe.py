"""
laserframe.py

This module defines the LaserFrame class, which is used to manage dynamically allocated data for agents or nodes/patches.
The LaserFrame class is similar to a database table or a Pandas DataFrame and supports scalar and vector properties.

Classes:
    LaserFrame: A class to manage dynamically allocated data for agents or nodes/patches.

Usage Example:
    ```python
    laser_frame = LaserFrame(capacity=100)
    laser_frame.add_scalar_property('age', dtype=np.int32, default=0)
    laser_frame.add_vector_property('position', length=3, dtype=np.float32, default=0.0)
    start, end = laser_frame.add(10)
    laser_frame.sort(np.arange(10)[::-1])
    laser_frame.squash(np.array([True, False, True, False, True, False, True, False, True, False]))
    ```

Attributes:
    count (int): The current count of active elements.
    capacity (int): The maximum capacity of the frame.

Note:
    Since count can be less than capacity, properties return slices of the underlying arrays up to count by default so users do not have to include the slice themselves.
    I.e., if `lf` is a LaserFrame, then `lf.age` returns `lf._age[0:lf.count]` automatically.
    The full underlying array is always available as `lf._age` (or whatever the property name is).
    The slice returned is valid for all NumPy operations, including assignment, as well as for use with Numba compiled functions.
"""

from functools import reduce
from operator import mul

import h5py
import numpy as np

from laser.core.utils import calc_capacity


class LaserFrame:
    """
    The LaserFrame class, similar to a db table or a Pandas DataFrame, holds dynamically
    allocated data for agents (generally 1-D or scalar) or for nodes|patches (e.g., 1-D for
    scalar value per patch or 2-D for time-varying per patch)."""

    def __init__(self, capacity: int, initial_count: int = -1, **kwargs):
        """
        Initialize a LaserFrame object.

        Parameters:
            capacity (int): The maximum capacity of the frame.
                            Must be a positive integer.
            initial_count (int): The initial number of active elements in the frame.
                                 Must be a positive integer <= capacity.
            **kwargs (dict): Additional keyword arguments to set as attributes of the object.

        Raises:
            ValueError: If capacity or initial_count is not a positive integer,
                        or if initial_count is greater than capacity.
        """
        if not isinstance(capacity, (int, np.integer)) or capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer, got {capacity}.")

        if initial_count == -1:
            initial_count = capacity

        if not isinstance(initial_count, (int, np.integer)) or initial_count < 0:
            raise ValueError(f"Initial count must be a non-negative integer, got {initial_count}.")

        if initial_count > capacity:
            raise ValueError(f"Initial count ({initial_count}) cannot exceed capacity ({capacity}).")

        self._count = initial_count
        self._capacity = capacity
        self._properties = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__class__.add_property = self.__class__.add_scalar_property  # alias

        return

    # dynamically add a property to the class
    def add_scalar_property(self, name: str, dtype=np.uint32, default=0) -> None:
        """
        Add a scalar property to the class.

        This method initializes a new scalar property for the class instance. The property is
        stored as a 1-D NumPy array (scalar / entry) with a specified data type and default value.

        Parameters:
            name (str): The name of the scalar property to be added.
            dtype (data-type, optional): The desired data type for the property. Default is np.uint32.
            default (scalar, optional): The default value for the property. Default is 0.

        Returns:
            None
        """

        # initialize the property to a NumPy array with of size self._capacity, dtype, and default value
        if hasattr(self, name):
            raise ValueError(f"Property '{name}' already exists in LaserFrame.")

        setattr(self, f"_{name}", np.full(self._capacity, default, dtype=dtype))
        self._properties[name] = getattr(self, f"_{name}")

        return

    def add_vector_property(self, name: str, length: int, dtype=np.uint32, default=0) -> None:
        """
        Adds a vector property to the object.

        This method initializes a new property with the given name as a 2-D NumPy array (vector per entry).

        The array will have a shape of (length, self._capacity) and will be filled
        with the specified default value. The data type of the array elements is
        determined by the `dtype` parameter.

        Parameters:
            name (str): The name of the property to be added.
            length (int): The length of the vector.
            dtype (data-type, optional): The desired data-type for the array, default is np.uint32.
            default (scalar, optional): The default value to fill the array with, default is 0.

        Returns:
            None
        """

        # initialize the property to a NumPy array with of size (length, self._capacity), dtype, and default value
        if hasattr(self, name):
            raise ValueError(f"Property '{name}' already exists in LaserFrame.")

        setattr(self, f"_{name}", np.full((length, self._capacity), default, dtype=dtype))
        self._properties[name] = getattr(self, f"_{name}")

        return

    def __getattr__(self, name: str):
        if name in self._properties:
            backing = self._properties[name]
            # Handle scalar and vector properties
            return backing[0 : self.count] if len(backing.shape) == 1 else backing[:, 0 : self.count]
        raise AttributeError(f"'LaserFrame' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if ("_properties" in self.__dict__) and (name in self._properties):
            raise RuntimeError(f"Cannot reassign property '{name}'. Modify the array in place instead, e.g., lf.{name}[:] = new_values")
        else:
            super().__setattr__(name, value)

    def add_array_property(self, name: str, shape: tuple, dtype=np.uint32, default=0) -> None:
        """
        Adds an array property to the object.

        This method initializes a new property with the given name as a multi-dimensional NumPy array.

        The array will have the given shape (note that there is no implied dimension of size self._capacity),
        datatype (default is np.uint32), and default value (default is 0).

        Parameters:
            name (str): The name of the property to be added.
            shape (tuple): The shape of the array.
            dtype (data-type, optional): The desired data-type for the array, default is np.uint32.
            default (scalar, optional): The default value to fill the array with, default is 0.

        Returns:
            None
        """

        if hasattr(self, name):
            raise ValueError(f"Property '{name}' already exists in LaserFrame.")

        # initialize the property to a NumPy array with given shape, dtype, and default value
        setattr(self, name, np.full(shape, default, dtype=dtype))
        return

    @property
    def count(self) -> int:
        """
        Returns the current count (equivalent to len()).

        Returns:
            int: The current count value.
        """

        return self._count

    @property
    def capacity(self) -> int:
        """
        Returns the capacity of the laser frame (total possible entries for dynamic properties).

        Returns:
            int: The capacity of the laser frame.
        """

        return self._capacity

    def add(self, count: int) -> tuple[int, int]:
        """
        Adds the specified count to the current count of the LaserFrame.

        This method increments the internal count by the given count, ensuring that the total does not exceed the frame's capacity. If the addition would exceed the capacity, an assertion error is raised. This method is typically used to add new births during the simulation.

        Parameters:
            count (int): The number to add to the current count.

        Returns:
            tuple[int, int]: A tuple containing the [start index, end index) after the addition.

        Raises:
            AssertionError: If the resulting count exceeds the frame's capacity.
        """

        if not self._count + count <= self._capacity:
            raise ValueError(f"frame.add() exceeds capacity ({self._count=} + {count=} > {self._capacity=})")

        i = self._count
        self._count += int(count)
        j = self._count
        return i, j

    def __len__(self) -> int:
        return self._count

    def sort(self, indices, verbose: bool = False) -> None:
        """
        Sorts the elements of the object's numpy arrays based on the provided indices.

        Parameters:
            indices (np.ndarray): An array of indices used to sort the numpy arrays. Must be of integer type and have the same length as the frame count (`self._count`).
            verbose (bool, optional): If True, prints the sorting progress for each numpy array attribute. Defaults to False.

        Raises:
            AssertionError: If `indices` is not an integer array or if its length does not match the frame count of active elements.
        """

        _is_instance(indices, np.ndarray, f"Indices must be a numpy array (got {type(indices)})")
        _has_shape(indices, (self._count,), f"Indices must have the same length as the frame active element count ({self._count})")
        _is_dtype(indices, np.integer, f"Indices must be an integer array (got {indices.dtype})")

        for name, data in self._properties.items():
            # Only sorting scalar properties at the moment.
            # TODO support sorting vector properties.
            if data.shape == (self._capacity,):
                if verbose:
                    print(f"Sorting {self._count:,} elements of {name} ... ", end="")
                temp = data.copy()
                data[0 : self._count] = temp[indices]
                if verbose:
                    print("done.")

        return

    def squash(self, indices, verbose: bool = False) -> None:
        """
        Reduces the active count of the internal numpy arrays keeping only elements True in the provided boolean indices.

        Parameters:
            indices (np.ndarray): A boolean array indicating which elements to keep. Must have the same length as the current frame active element count.
            verbose (bool, optional): If True, prints detailed information about the squashing process. Defaults to False.

        Raises:
            AssertionError: If `indices` is not a boolean array or if its length does not match the current frame active element count.

        Returns:
            None
        """

        _is_instance(indices, np.ndarray, f"Indices must be a numpy array (got {type(indices)})")
        _has_shape(indices, (self._count,), f"Indices must have the same length as the frame active element count ({self._count})")
        _is_dtype(indices, np.bool_, f"Indices must be a boolean array (got {indices.dtype})")

        current_count = self._count
        selected_count = indices.sum()
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 1 and value.shape[0] == self._capacity:
                if verbose:
                    print(f"Squashing {key} from {current_count:,} to {selected_count:,}")
                value[:selected_count] = value[:current_count][indices]
        self._count = selected_count

        return

    def save_snapshot(self, path, results_r=None, pars=None):
        """
        Save this LaserFrame and optional extras to an HDF5 snapshot file.

        Parameters:
            path (Path): Destination file path
            results_r (np.ndarray): Optional 2D numpy array of recovered counts
            pars (PropertySet or dict): Optional PropertySet or dict of parameters
        """
        from laser.core.propertyset import PropertySet  # to avoid circular import

        with h5py.File(path, "w") as f:
            self._save(f, "people")

            if results_r is not None:
                f.create_dataset("recovered", data=results_r)

            if pars is not None and isinstance(pars, (dict, PropertySet)):
                data = pars.to_dict() if isinstance(pars, PropertySet) else pars
                self._save_dict(data, f.create_group("pars"))

    def _save(self, parent_group, name):
        """
        Internal method to save this LaserFrame under the given group name.
        """
        group = parent_group.create_group(name)
        group.attrs["count"] = self._count
        group.attrs["capacity"] = self._capacity

        for name, data in self._properties.items():
            # Currently only saving scalar properties (implied by loading logic)
            if data.shape == (self._capacity,):
                group.create_dataset(name, data=data[0 : self._count])

        return

    def _save_dict(self, data, group):
        """
        Internal method to save a dict as datasets and attributes in a group.
        """
        for key, value in data.items():
            try:
                group.create_dataset(key, data=value)
            except TypeError:
                group.attrs[key] = str(value)

    @classmethod
    def load_snapshot(cls, path, cbr, nt):
        """
        Load a LaserFrame and optional extras from an HDF5 snapshot file.
        This function assumes the snapshot contains per-agent node_id,
        and that recovered counts are stored in (time, node) layout.

        Args:
            path (str): Path to the HDF5 snapshot file.
            cbr (float or array-like, optional): Crude birth rate (per 1000/year).
                Must be provided together with nt, or both must be None.
                If provided, capacity is calculated to accommodate expected growth.
                If None, capacity is set to current count (no growth expected).
            nt (int, optional): Simulation duration (number of ticks).
                Must be provided together with cbr, or both must be None.

        Returns:
            frame (LaserFrame): Loaded LaserFrame object.
            results_r (np.ndarray or None): Optional 2D numpy array of recovered counts.
            pars (dict or None): Optional dictionary of parameters.

        Raises:
            ValueError: If only one of cbr or nt is provided.
        """

        with h5py.File(path, "r") as f:
            group = f["people"]
            count = int(group.attrs["count"])

            # Load parameters first
            if "pars" in f:
                pars_group = f["pars"]
                pars = {
                    key: (pars_group[key][()].decode() if isinstance(pars_group[key][()], bytes) else pars_group[key][()])
                    for key in pars_group
                }
                pars.update({key: (val.decode() if isinstance(val, bytes) else val) for key, val in pars_group.attrs.items()})
            else:
                pars = {}

            # Validate that cbr and nt are both provided or both None
            if (cbr is None) != (nt is None):
                raise ValueError("cbr and nt must both be provided or both be None. " "Cannot calculate capacity with only one parameter.")

            # Compute capacity if values are provided
            if cbr is not None and nt is not None:
                recovered = f["recovered"][()] if "recovered" in f else None

                if "node_id" not in group:
                    raise ValueError("Snapshot is missing 'node_id'; cannot determine per-node population.")

                node_ids = group["node_id"][:count]

                if len(node_ids) != count:
                    raise ValueError(f"node_id array length ({len(node_ids)}) does not match count ({count})")

                nnodes = int(node_ids.max()) + 1

                n_ppl = np.bincount(node_ids, minlength=nnodes).astype(np.int32)

                # Add recovered counts from initial timestep (t=0)
                if recovered is not None:
                    if recovered.shape[1] != nnodes:
                        raise ValueError(f"Recovered node count ({recovered.shape[1]}) does not match inferred node count ({nnodes})")
                    n_ppl += recovered[0, :].astype(np.int32)

                # We choose to allow:
                # cbr = [[30, 30], [30, 30]]        # list of lists
                # cbr = np.matrix(...)             # ndarray subclass
                # cbr = pandas.DataFrame(...)
                # cbr = xarray.DataArray(...)
                # cbr = np.asarray(cbr)

                if cbr.ndim != 2:
                    raise ValueError("load_snapshot requires cbr to be a 2-D array of shape " "(num_timesteps, num_nodes)")

                if cbr.shape[0] != nt:
                    raise ValueError(f"CBR time dimension ({cbr.shape[0]}) does not match nt ({nt})")

                if cbr.shape[1] != nnodes:
                    raise ValueError(f"CBR node dimension ({cbr.shape[1]}) does not match inferred node count ({nnodes})")

                if n_ppl.shape[0] != nnodes:
                    raise ValueError(f"Inferred population length ({n_ppl.shape[0]}) does not match expected node count ({nnodes})")

                estimate = calc_capacity(
                    birthrates=cbr,
                    initial_pop=n_ppl,
                    safety_factor=1.0,
                )
                capacity = int(estimate.sum())
            else:
                capacity = count

            if capacity < count:
                raise ValueError(f"There is no way capacity ({capacity}) should ever be less than count ({count}).")

            # Now construct frame
            frame = cls(capacity=capacity, initial_count=count)
            for key in group:
                data = group[key][:]
                dtype = data.dtype
                frame.add_scalar_property(name=key, dtype=dtype, default=0)
                getattr(frame, key)[:count] = data

            results_r = f["recovered"][()] if "recovered" in f else None

        return frame, results_r, pars

    def describe(self, target=None) -> str:
        """
        Return a formatted string description of the laserframe object, including its attributes and their values.

        Args:
            target (string): Optional string for the report header (generally the name of the LaserFrame variable, e.g., "People". Unlike functions, we can't get the name of a variable automatically).

        Returns:
            str: A formatted string describing the laserframe object, including its capacity, count, and
                 details of its scalar, vector, and other properties.
        """
        description = []

        description.append("")
        if target:
            description.append(f"Laserframe Report for `{target}`:")
        description.append(f"Capacity: {self.capacity:>13,}")
        description.append(f"Count:    {self.count:>13,}")
        description.append("")

        scalars = []
        vectors = []
        others = []
        # Look through all attributes of the laserframe object and collect information about numpy arrays
        # to print a report of their sizes and types. This includes scalars, vectors, and other arrays.
        # Scalars are 1-D arrays with shape (capacity,)
        # Vectors are 2-D arrays with shape (length, capacity),
        # Others are arrays with different shapes.
        # for attr_name in sorted(dir(self)):
        for attr_name in sorted(self.__dict__.keys()):
            attr = getattr(self, attr_name)
            if isinstance(attr, np.ndarray):
                attr_name = attr_name.lstrip("_")  # remove leading underscore if present
                if attr.shape == (self.capacity,):
                    # name, dtype, individual size, allocated size, in-use size
                    scalars.append((attr_name, attr.dtype.name, attr.dtype.itemsize, attr.nbytes, attr.dtype.itemsize * self.count))
                elif attr.shape == (attr.shape[0], self.capacity):
                    # name, dtype, count, individual size, allocated size, in-use size
                    i = attr.dtype.itemsize * attr.shape[0]
                    u = i * self.count
                    vectors.append((attr_name, attr.dtype.name, attr.shape[0], i, attr.nbytes, u))
                else:
                    # name, dtype, individual size, shape, num_elements, allocated size
                    others.append((attr_name, attr.dtype.name, attr.dtype.itemsize, attr.shape, reduce(mul, attr.shape, 1), attr.nbytes))

        def sep(width, char="-"):
            string = char * width

            return string

        def header(name, width):
            string = "\n".join([sep(width, "="), f"{name:^{width}}", sep(width, "=")])

            return string

        def row(info):
            string = " | ".join(
                f"{data:{align}{width},}" if not isinstance(data, str) else f"{data:{align}{width}}" for data, align, width in info
            )

            return string

        if scalars:
            # Determine the width for the name column based on the longest name
            nwidth = max(len(name) for name, *_ in scalars)

            # Datatype, individual size, allocated size, in-use size, and total column widths
            dwidth = 9
            iwidth = 23
            awidth = 22
            uwidth = 20
            twidth = nwidth + 3 + dwidth + 3 + iwidth + 3 + awidth + 3 + uwidth

            description.append(header("Scalars", twidth))
            description.append(
                row(
                    [
                        ("Name", "<", nwidth),
                        ("Datatype", "^", dwidth),
                        ("Individual Size (bytes)", "^", iwidth),
                        ("Allocated Size (bytes)", ">", awidth),
                        ("In Use Size (bytes)", ">", uwidth),
                    ]
                )
            )
            description.append(sep(twidth))

            # For each scalar, accumulate the individual size, allocated size, and in-use size
            individual = 0
            allocated = 0
            in_use = 0
            for name, data_type, individual_size, allocated_size, in_use_size in scalars:
                individual += individual_size
                allocated += allocated_size
                in_use += in_use_size
                description.append(
                    row(
                        [
                            (name, "<", nwidth),
                            (data_type, "^", dwidth),
                            (individual_size, "^", iwidth),
                            (allocated_size, ">", awidth),
                            (in_use_size, ">", uwidth),
                        ]
                    )
                )

            description.append(sep(twidth))
            description.append(
                row([("Total", "<", nwidth), ("", "^", dwidth), (individual, "^", iwidth), (allocated, ">", awidth), (in_use, ">", uwidth)])
            )
            description.append(sep(twidth))
            description.append("")

        if vectors:
            # Determine the width for the name column based on the longest name
            nwidth = max(len(name) for name, *_ in vectors)

            # Datatype, count, individual size, allocated size, in-use size, and total column widths
            dwidth = 9
            cwidth = 6
            iwidth = 23
            awidth = 22
            uwidth = 20
            twidth = nwidth + 3 + dwidth + 3 + cwidth + 3 + iwidth + 3 + awidth + 3 + uwidth

            description.append(header("Vectors", twidth))
            description.append(
                row(
                    [
                        ("Name", "<", nwidth),
                        ("Datatype", "^", dwidth),
                        ("Count", "^", cwidth),
                        ("Individual Size (bytes)", "^", iwidth),
                        ("Allocated Size (bytes)", ">", awidth),
                        ("In Use Size (bytes)", ">", uwidth),
                    ]
                )
            )
            description.append(sep(twidth))

            # For each vector, accumulate the individual size, allocated size, and in-use size
            individual = 0
            allocated = 0
            in_use = 0
            for name, data_type, count, individual_size, allocated_size, in_use_size in vectors:
                individual += individual_size
                allocated += allocated_size
                in_use += in_use_size
                description.append(
                    row(
                        [
                            (name, "<", nwidth),
                            (data_type, "^", dwidth),
                            (count, "^", cwidth),
                            (individual_size, "^", iwidth),
                            (allocated_size, ">", awidth),
                            (in_use_size, ">", uwidth),
                        ]
                    )
                )

            description.append(sep(twidth))
            description.append(
                row(
                    [
                        ("Total", "<", nwidth),
                        ("", "^", dwidth),
                        ("", "^", cwidth),
                        (individual, "^", iwidth),
                        (allocated, ">", awidth),
                        (in_use, ">", uwidth),
                    ]
                )
            )
            description.append(sep(twidth))
            description.append("")

        if others:
            # Determine the width for the name column based on the longest name
            nwidth = max(len(name) for name, *_ in others)

            # Datatype, individual size, shape, num_elements, allocated size, and total column widths
            dwidth = 8
            iwidth = 23
            swidth = 15
            cwidth = 12
            awidth = 22
            twidth = nwidth + 3 + dwidth + 3 + iwidth + 3 + swidth + 3 + cwidth + 3 + awidth

            description.append(header("Others", twidth))
            description.append(
                row(
                    [
                        ("Name", "<", nwidth),
                        ("Datatype", "^", dwidth),
                        ("Individual Size (bytes)", "^", iwidth),
                        ("Shape", "^", swidth),
                        ("Num Elements", "^", cwidth),
                        ("Allocated Size (bytes)", ">", awidth),
                    ]
                )
            )
            description.append(sep(twidth))

            # For each other attribute, accumulate the individual size, allocated size, and in-use size
            individual = 0
            allocated = 0
            for name, data_type, individual_size, shape, num_elements, allocated_size in others:
                individual += individual_size
                allocated += allocated_size
                description.append(
                    row(
                        [
                            (name, "<", nwidth),
                            (data_type, "^", dwidth),
                            (individual_size, "^", iwidth),
                            (str(shape), "^", swidth),
                            (num_elements, "^", cwidth),
                            (allocated_size, ">", awidth),
                        ]
                    )
                )

            description.append(sep(twidth))
            description.append(
                row(
                    [
                        ("Total", "<", nwidth),
                        ("", "^", dwidth),
                        (individual, "^", iwidth),
                        ("", "^", swidth),
                        ("", "^", cwidth),
                        (allocated, ">", awidth),
                    ]
                )
            )
            description.append(sep(twidth))
            description.append("")

        description = "\n".join(description)

        return description


# Sanity checks


def _is_instance(obj, types, message):
    if not isinstance(obj, types):
        raise TypeError(message)

    return


# def _has_dimensions(obj, dimensions, message):
#     if not len(obj.shape) == dimensions:
#         raise TypeError(message)

#     return


def _is_dtype(obj, dtype, message):
    if not np.issubdtype(obj.dtype, dtype):
        raise TypeError(message)

    return


# def _has_values(check, message):
#     if not np.all(check):
#         raise ValueError(message)

#     return


def _has_shape(obj, shape, message):
    if not obj.shape == shape:
        raise TypeError(message)

    return

"""SortedQueue implementation using NumPy and Numba."""

from functools import lru_cache
from typing import Any

import numba as nb
import numpy as np


class SortedQueue:
    """Fixed-capacity priority queue over indices into an external value array.

    The queue stores `np.uint32` *indices* into a user-supplied `values` array; pushes
    and pops compare via `values[index]` so the underlying value array is never copied,
    only indexed. This is the data structure LASER models use to schedule per-agent
    events (deaths, transitions, exogenous arrivals) without per-event allocation —
    designed for cohorts of tens to hundreds of millions of agents.

    The implementation mirrors the algorithm in CPython's `heapq` module, with the
    sift-forward and sift-backward kernels compiled per `values.dtype` via Numba
    `@nb.njit` (cached through `lru_cache`).

    Attributes:
        indices (np.ndarray of np.uint32): Backing array of length `capacity` holding
            heap-ordered indices into `values`. Only the first `size` entries are valid.
        values (np.ndarray): The external array of sort keys, supplied at construction
            and aliased (not copied). The queue assumes these values do not move; if you
            mutate `values[i]` while `i` is in the queue, you must re-heapify yourself.
        size (np.uint32): Number of indices currently in the queue (`0 <= size <= capacity`).

    Raises:
        IndexError: If `push` is called when `size == capacity`, or if `pop` is called
            when `size == 0`.

    **Example**:

        import numpy as np
        from laser.core import SortedQueue

        priorities = np.array([5.0, 1.0, 3.0, 7.0], dtype=np.float32)
        sq = SortedQueue(capacity=4, values=priorities)
        for i in range(4):
            sq.push(i)
        idx, val = sq.popiv()  # idx=1, val=1.0 (smallest)
    """

    # https://github.com/python/cpython/blob/5592399313c963c110280a7c98de974889e1d353/Modules/_heapqmodule.c
    # https://github.com/python/cpython/blob/5592399313c963c110280a7c98de974889e1d353/Lib/heapq.py

    def __init__(self, capacity: int, values: np.ndarray):
        """
        Initializes a new instance of the class with a specified capacity and reference to existing, sortable values.

        This implementation is specific to LASER and the expectation of tracking 10s or 100s of millions of agents.

        We expect the sortable (or priority) values to already be in a NumPy array, usually a property of a LaserFrame object.

        The `push()` and `pop()` will take _indices_ into this array and will sort on values[i].
        This avoids making copies of the sort values.

        Parameters:
            capacity (int): The maximum number of elements the queue can hold.
            values (np.ndarray): A reference to an array of values to be accessed by the queue.
        """

        self.indices = np.zeros(capacity, dtype=np.uint32)
        self.values = values
        self.size = np.uint32(0)

        self._siftforward, self._siftbackward = _make_sifts(values.dtype)

        return

    def push(self, index) -> None:
        """
        Insert an element into the sorted queue.

        This method adds an element at the back of the sorted queue and then
        ensures the heap property is maintained by sifting the element forward
        to its correct position.

        Parameters:
            index (int): The index of the element to be added to the sorted queue.

        Raises:
            IndexError: If the sorted queue is full.
        """

        if self.size >= len(self.indices):
            raise IndexError("Sorted queue is full")
        self.indices[self.size] = index
        self._siftforward(self.indices, self.values, np.uint32(0), self.size)
        self.size += np.uint32(1)
        return

    def peeki(self) -> np.uint32:
        """
        Returns the index of the smallest value element in the sorted queue without removing it.

        Raises:
            IndexError: If the sorted queue is empty.

        Returns:
            np.uint32: The index of the smallest value element.
        """

        if self.size == 0:
            raise IndexError("Sorted queue is empty")
        return self.indices[0]

    def peekv(self) -> Any:
        """
        Return the smallest value from the sorted queue without removing it.

        Raises:
            IndexError: If the sorted queue is empty.

        Returns:
            Any: The value with the smallest value in the sorted queue.
        """

        if self.size == 0:
            raise IndexError("Sorted queue is empty")
        return self.values[self.indices[0]]

    def peekiv(self) -> tuple[np.uint32, Any]:
        """
        Returns the index and value of the smallest value element in the sorted queue without removing it.

        Returns:
            tuple[np.uint32, Any]: A tuple containing the index and value of the smallest value element.

        Raises:
            IndexError: If the sorted queue is empty.
        """

        if self.size == 0:
            raise IndexError("Sorted queue is empty")
        return (self.indices[0], self.values[self.indices[0]])

    def popi(self) -> np.uint32:
        """
        Removes and returns the index of the smallest value element in the sorted queue.

        This method first retrieves the index of the smallest value element using `peeki()`,
        then removes the element from the queue using `pop()`, and finally returns the index.

        Returns:
            np.uint32: The index of the smallest value element in the sorted queue.
        """

        index = self.peeki()
        self.__pop()

        return index

    def popv(self) -> Any:
        """
        Removes and returns the value at the front of the sorted queue.

        This method first retrieves the value at the front of the queue without
        removing it by calling `peekv()`, and then removes the front element
        from the queue by calling `pop()`. The retrieved value is then returned.

        Returns:
            Any: The value at the front of the sorted queue.
        """

        value = self.peekv()
        self.__pop()

        return value

    def popiv(self) -> tuple[np.uint32, Any]:
        """
        Removes and returns the index and value of the smallest value element in the sorted queue.

        This method first retrieves the index and value of the smallest value element using `peekiv()`,
        then removes the element from the queue using `pop()`, and finally returns the index and value.

        Returns:
            tuple[np.uint32, Any]: A tuple containing the index and value of the smallest value element.
        """
        ivtuple = self.peekiv()
        self.__pop()

        return ivtuple

    def __pop(self) -> None:
        """
        Removes the smallest value element from the sorted queue.

        Raises:
            IndexError: If the sorted queue is empty.

        Side effects:
            - Decreases the size of the sorted queue by one.
            - Reorganizes the internal structure of the sorted queue to maintain the heap property.
        """

        if self.size == 0:
            raise IndexError("Priority queue is empty")
        self.size -= np.uint32(1)
        self.indices[0] = self.indices[self.size]
        self._siftbackward(self.indices, self.values, np.uint32(0), self.size)
        return

    def __len__(self) -> int:
        """
        Return the number of elements in the sorted queue.

        Returns:
            int: The number of elements in the sorted queue.
        """

        return int(self.size)


@lru_cache(maxsize=10)  # 4 signed ints, 4 unsigned ints, 2 floats
def _make_sifts(npdtype):
    np_nb_map = {
        np.float32(42).dtype: nb.float32[:],
        np.float64(42).dtype: nb.float64[:],
        np.uint8(42).dtype: nb.uint8[:],
        np.uint16(42).dtype: nb.uint16[:],
        np.uint32(42).dtype: nb.uint32[:],
        np.uint64(42).dtype: nb.uint64[:],
        np.int8(42).dtype: nb.int8[:],
        np.int16(42).dtype: nb.int16[:],
        np.int32(42).dtype: nb.int32[:],
        np.int64(42).dtype: nb.int64[:],
    }

    nbdtype = np_nb_map[npdtype]

    @nb.njit((nb.uint32[:], nbdtype, nb.uint32, nb.uint32), nogil=True)
    def _siftforward(indices, values, startpos, pos):  # pragma: no cover
        inewitem = indices[pos]
        vnewitem = values[inewitem]
        # Follow the path to the root, moving parents backward until finding a place newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            iparent = indices[parentpos]
            vparent = values[iparent]
            if vnewitem < vparent:
                indices[pos] = iparent
                pos = parentpos
                continue
            break
        indices[pos] = inewitem

        return

    @nb.njit((nb.uint32[:], nbdtype, nb.uint32, nb.uint32), nogil=True)
    def _siftbackward(indices, values, pos, size):  # pragma: no cover
        endpos = size
        startpos = pos
        inewitem = indices[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not values[indices[childpos]] < values[indices[rightpos]]:
                childpos = rightpos
            # Move the smaller child up.
            indices[pos] = indices[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents forward).
        indices[pos] = inewitem
        _siftforward(indices, values, startpos, pos)
        return

    return _siftforward, _siftbackward

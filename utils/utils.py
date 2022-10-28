"""Some utilities"""
import numpy as np
from itertools import repeat, accumulate
from operator import mul

def is_iterable(obj):
    """Check if `obj` is iterable"""
    return hasattr(obj, "__iter__")


def sanitize(A):
    """Set to zero all values of A under the machine epsilon"""
    return A * (np.abs(A) > np.finfo(A.dtype).eps * 10)


def multiindex(index, sizes, length, offsets=0):
    """
    Construct multiindex tuple given a single index `index`

    Parameters
    ----------
    index: int
        the index to convert
    sizes: int or iterable(int)
        the range size of each sub-index of the multiindex
    lenght: int
        the number of sub-indices of the multiindex
    offset: int or iterable(int)
        the offset to apply to each sub-index

    Return
    ----------
    tuple(int)
    """
    offsets = repeat(offsets, length) if not is_iterable(offsets) else offsets
    sizes = repeat(sizes, length) if not is_iterable(sizes) else sizes
    prod = accumulate(sizes, mul, initial=1)
    return tuple((index // p) % s + o for p, s, o in zip(prod, sizes, offsets))
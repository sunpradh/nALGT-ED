"""Some utilities"""
import numpy as np
import pickle
from collections.abc import Iterable
from itertools import repeat, accumulate, product
from functools import reduce
from operator import mul

from group import Irreps
from utils.mytyping import IrrepIndex

def unpickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def is_iterable(obj):
    """Check if `obj` is iterable"""
    return hasattr(obj, "__iter__")


def sanitize(A):
    """Set to zero all values of A under the machine epsilon"""
    return A * (np.abs(A) > np.finfo(A.dtype).eps * 10)


def multiply(iterable: Iterable):
    """Multiply the elements of an iterable"""
    return reduce(mul, iterable, 1.0)


def all_true(iterable: Iterable):
    """Check if all the elements of the iterable are true"""
    return bool(multiply(iterable))


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


def iter_irrep_mels(irreps: Irreps, irr_conf: Iterable[IrrepIndex]):
    """
    Returns an iterator overr all the possible sequences of mel indices
    (j1, m1, n1, j2, m2, n2, ...) given a sequence of irreps (j1, j2, ...)
    """
    indices_list = [
        list(
            product([j], range(irreps.dim(j)), range(irreps.dim(j)))
        )
        for j in irr_conf
    ]
    return product(*indices_list)

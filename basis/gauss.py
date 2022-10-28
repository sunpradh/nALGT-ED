"""
Gauss operators in the irrep basis
"""
import numpy as np
from functools import reduce

from group import Group_elem
from utils.utils import sanitize
from utils.mytyping import IrrepFn

def left_irrep(g: Group_elem, irrep):
    return np.conj(irrep(g)).T

def right_irrep(g: Group_elem, irrep):
    return irrep(g)

def gauss_operator(
        g: Group_elem,
        irrep_seq: tuple[IrrepFn, ...],
        sanitized=False
    ) -> np.ndarray:
    """
    Construct the gauss operator on a vertex, for a given group element `g`
    and irrep configuration `irrep_sep` on the links of the vertex
    """
    actions = (left_irrep, left_irrep, right_irrep, right_irrep)
    gauss = reduce(np.kron, [
                   action(g, irrep)
                       for action, irrep in zip(actions, irrep_seq)
                ])
    return sanitize(gauss) if sanitized else gauss

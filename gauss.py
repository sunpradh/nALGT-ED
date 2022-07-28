import numpy as np

from functools import reduce
from utils import sanitize
from group import Group_elem

def left_irrep(g: Group_elem, irrep):
    return np.conj(irrep(g)).T

def right_irrep(g: Group_elem, irrep):
    return irrep(g)

def gauss_operator(g: Group_elem, irrep_seq: tuple, sanitized=False):
    actions = (left_irrep, left_irrep, right_irrep, right_irrep)

    gauss = reduce(np.kron, [
                   action(g, irrep)
                       for action, irrep in zip(actions, irrep_seq)
                ])
    return sanitize(gauss) if sanitized else gauss

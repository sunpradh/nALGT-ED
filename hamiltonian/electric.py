"""
Compute the electric Hamiltonian
"""

import scipy.sparse as sparse
from collections.abc import Callable, Iterable
from tqdm import tqdm

from basis.basis import Basis
from group import Group_elem, Irreps


def elec_single_link_fn(
        generating_set: Iterable[Group_elem],
        irreps: Irreps
    ) -> Callable[[int], float]:
    def f(j):
        dim = irreps.dim(j)
        char_sum = sum(irreps.chars[j](g) for g in generating_set)
        return len(generating_set) - (char_sum / dim)
    return f


def elec_hamiltonian(
        basis: Basis,
        generating_set: list[Group_elem],
        irreps: Irreps,
        progress_bar = False
    ) -> list[float]:
    f = elec_single_link_fn(generating_set, irreps)
    n_states = len(basis.states)
    H = sparse.dok_matrix((n_states, n_states))
    iterator = tqdm(range(n_states)) if progress_bar else range(n_states)
    for row in iterator:
        state = basis.states[row]
        H[row, row] = sum(f(j) for j in state[0])
    return H

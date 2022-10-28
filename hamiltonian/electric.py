"""
Compute the electric Hamiltonian
"""

from collections.abc import Callable, Iterable

from basis.basis import Basis
from group import Group_elem, Irreps


def electric_single_link_fn(
        generating_set: Iterable[Group_elem],
        irreps: Irreps
    ) -> Callable[[int], float]:
    def f(j):
        dim = irreps.dim(j)
        char_sum = sum(irreps.chars[j](g) for g in generating_set)
        return len(generating_set) - (char_sum / dim)
    return f


def electric_hamiltonian(
        basis: Basis,
        generating_set: list[Group_elem],
        irreps: Irreps
    ) -> list[float]:
    f = electric_single_link_fn(generating_set, irreps)
    electric_values = [
        sum(f(j) for j in state[0]) for state in basis.states
    ]
    return electric_values


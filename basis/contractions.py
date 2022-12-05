"""
Utilities for composing and contracting the gauge-invariant states
"""

import numpy as np
from itertools import chain
from functools import reduce

from basis import State, Basis
from utils.mytyping import PlaqVertices

Tensor = np.ndarray

# TODO aggiungere tests


def compose_inv_tensors(psi0: Tensor, psi1: Tensor) -> Tensor:
    final_shape = tuple(chain(psi0.shape, psi1.shape))
    result = np.kron(psi0, psi1).reshape(final_shape)
    return result


def tensor_around_plaq(
        basis: Basis,
        state: State,
        plaq:  PlaqVertices
    ) -> Tensor:
    return reduce(compose_inv_tensors, (basis(state)[v] for v in plaq))


def contract_outside_plaq(
        bra_tensor: Tensor,
        ket_tensor: Tensor
    ) -> Tensor:
    out_axes = [2, 3, 4, 7, 8, 9, 13, 14]
    return np.tensordot(
        np.conj(bra_tensor),
        ket_tensor,
        axes=(out_axes, out_axes)
    )


def contract_magnetic_elem(
        plaq_tensor: Tensor,
        bra_tensor: Tensor,
        ket_tensor: Tensor
    ) -> float | np.complex:
    axes_list1 = list(range(16))
    axes_list2 = [0, 6, 2, 1, 5, 3, 4, 7, 8, 14, 10, 9, 13, 11, 12, 15]
    psi_plaq = contract_outside_plaq(bra_tensor, ket_tensor)
    return np.tensordot(
        plaq_tensor,
        psi_plaq,
        axes = (axes_list2, axes_list1)
    )

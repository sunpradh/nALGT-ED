"""
Precompute the representation-theoretic coeffients of the
magnetic terms for a plaquette
"""

# TODO: OPTIMIZE OPTIMIZE OPTIMIZE

import numpy as np

from itertools import product
from functools import reduce, cache
from operator import mul

from group import Group, Group_elem, Irreps
from utils import sanitize

# Index format in the irrep basis (j, m, n)
MelIndex = tuple[int, int, int] # (j, m, n)
# Index format for a plaquette state (j_1, m_1, n_1; ...; j_4, m_4, n_4)
PlaqIndex = tuple[MelIndex, MelIndex, MelIndex, MelIndex]

def multiply(iterable):
    return reduce(mul, iterable)

@cache
def prefactor(
        group: Group,
        irreps: Irreps,
        plaq_ket: PlaqIndex,
        plaq_bra: PlaqIndex
    ):
    """Compute the prefactor"""
    ord_group = len(group)
    irrep_ket_dim = multiply(irreps.dim(i[0]) for i in plaq_ket)
    irrep_bra_dim = multiply(irreps.dim(i[0]) for i in plaq_bra)
    return 2 * np.sqrt(irrep_ket_dim * irrep_bra_dim) / (ord_group ** 4)


@cache
def plaq_character(
        irreps: Irreps,
        g_elems: tuple[Group_elem, ...],
        magn_irrep: int
    ):
    g1, g2, g3, g4 = g_elems
    return np.real(irreps.chars[magn_irrep](g1 * g2 * (~g3) * (~g4)))


@cache
def plaq_mels(
        irreps: Irreps,
        g_elems: tuple[Group_elem, ...],
        plaq_state: PlaqIndex
    ):
    return multiply(irreps.mel(*jmn)(g) for jmn, g in zip(plaq_state, g_elems))


def wl_sum_term(
        irreps: Irreps,
        g_elems: tuple[Group_elem, ...],
        plaq_ket: PlaqIndex,
        plaq_bra: PlaqIndex,
        magn_irrep: int
    ):
    return \
        plaq_character(irreps, g_elems, magn_irrep) * \
        plaq_mels(irreps, g_elems, plaq_ket) * \
        np.conj(plaq_mels(irreps, g_elems, plaq_bra))


def wl_mel(
        group: Group,
        irreps: Irreps,
        plaq_ket: PlaqIndex,
        plaq_bra: PlaqIndex,
        magn_irrep: int,
        group_range: list[tuple[Group_elem, ...]] or None = None
    ):
    if not group_range:
        group_range = product(group, repeat=4)
    return \
        prefactor(group, irreps, plaq_ket, plaq_bra) * \
        sum(
            wl_sum_term(irreps, g_elems, plaq_ket, plaq_bra, magn_irrep)
            for g_elems in group_range
        )
    # For D4, it sums over 8^4 = 4096 elements
    # Not too efficient

def non_zero_plaq_char(group: Group, irreps: Irreps, magn_irrep: int):
    return [
        g_elems
        for g_elems in product(group, repeat=4)
        if sanitize(plaq_character(irreps, g_elems, magn_irrep))
    ]


def wl_matrix(
        group: Group,
        irreps: Irreps,
        magn_irrep: int
    ):
    irrep_inds = irreps.mel_indices()
    irreps_ket = product(irrep_inds, repeat=4)
    irreps_bra = product(irrep_inds, repeat=4)
    # Ranges over only a subset of G x G x G x G,
    # for which the character chi(g1 * g2 * ~g3 * ~g4) is non zero
    # when summing for a single matrix element
    g_range = non_zero_plaq_char(group, irreps, magn_irrep)
    return {
        bra: {
            ket: wl_mel(group, irreps, ket, bra, magn_irrep, g_range)
            for ket in irreps_ket
        } for bra in irreps_bra
    }

    # For D4, it iterates over 8^4 x 8^4 = 2^24 ~ 16mln elements
    # Not too efficient
    # Estimated time for the D4 case: 18.5 day

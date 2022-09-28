"""
Precompute the representation-theoretic coeffients of the
magnetic terms for a plaquette
"""

# TODO: OPTIMIZE OPTIMIZE OPTIMIZE

import numpy as np
import logging as log

from itertools import product
from functools import reduce, cache
from operator import mul
from pathos.multiprocessing import ProcessingPool as Pool

from group import Group, Group_elem, Irreps
from utils import sanitize

# Type hintings:
# - Index format in the irrep basis (j, m, n)
MelIndex = tuple[int, int, int]
# - Index format for a plaquette state (j_1, m_1, n_1; ...; j_4, m_4, n_4)
PlaqIndex = tuple[MelIndex, MelIndex, MelIndex, MelIndex]
# - Group tuple (g_1, g_2, g_3, g_4)
GroupTuple = tuple[Group_elem, Group_elem, Group_elem, Group_elem]

def multiply(iterable):
    return reduce(mul, iterable)

@cache
def prefactor(
        group: Group,
        irreps: Irreps,
        plaq_ket: PlaqIndex,
        plaq_bra: PlaqIndex
    ) -> float:
    """Compute the prefactor"""
    ord_group = len(group)
    irrep_ket_dim = multiply(irreps.dim(i[0]) for i in plaq_ket)
    irrep_bra_dim = multiply(irreps.dim(i[0]) for i in plaq_bra)
    return 2 * np.sqrt(irrep_ket_dim * irrep_bra_dim) / (ord_group ** 4)


@cache
def plaq_character(
        irreps: Irreps,
        g_elems: GroupTuple,
        magn_irrep: int
    ) -> float:
    g1, g2, g3, g4 = g_elems
    return np.real(irreps.chars[magn_irrep](g1 * g2 * (~g3) * (~g4)))


@cache
def plaq_mels(
        irreps: Irreps,
        g_elems: GroupTuple,
        plaq_state: PlaqIndex
    ) -> float | complex:
    """Computes
    $[\pi^{j_1}(g_1)]_{m_1 n_1} * ... * [\pi^{j_4}(g_4)]_{m_4 n_4}$
    """
    return multiply(irreps.mel(*jmn)(g) for jmn, g in zip(plaq_state, g_elems))


def wl_sum_term(
        irreps: Irreps,
        g_elems: GroupTuple,
        plaq_ket: PlaqIndex,
        plaq_bra: PlaqIndex,
        magn_irrep: int
    ) -> float | complex:
    """Computes one term of the sum for one matrix element of the wilson loop
    $$
        \Re\chi(g_1 g_2 g_3^{-1} g_4^{-1}) * \
        [\pi^{j_1}(g_1)]_{m_1 n_1} * ... * [\pi^{j_4}(g_4)]_{m_4 n_4} * \
        [\pi^{j'_1}(g_1)]^*_{m'_1 n'_1} * ... * [\pi^{j'_4}(g_4)]^*_{m'_4 n'_4}
    $$
    """
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
        group_range: list[GroupTuple] or None = None
    ) -> float | complex:
    """Computes a single matrix element of the Wilson loop"""
    if not group_range:
        group_range = product(group, repeat=4)
    mel = sanitize(
            prefactor(group, irreps, plaq_ket, plaq_bra) * \
            sum(
                wl_sum_term(irreps, g_elems, plaq_ket, plaq_bra, magn_irrep)
                for g_elems in group_range
            )
        )
    if mel:
        log.info(f'bra: {plaq_bra}, ket: {plaq_ket}, mel: {mel}')
    return mel
    # For D4
    #  - with full range it sums over 8^4 = 4096 elements
    #  - with non_zero_plaq_char it sums over 1024 elements
    # it is some improvement


def non_zero_plaq_char(group: Group, irreps: Irreps, magn_irrep: int):
    return [
        g_elems
        for g_elems in product(group, repeat=4)
        if sanitize(plaq_character(irreps, g_elems, magn_irrep))
    ]


def wl_matrix(
        group: Group,
        irreps: Irreps,
        magn_irrep: int,
    ):
    irrep_inds = irreps.mel_indices()
    irreps_bra = product(irrep_inds, repeat=4)
    # Ranges over only a subset of G x G x G x G,
    # for which the character chi(g1 * g2 * ~g3 * ~g4) is non zero
    # when summing for a single matrix element
    group_range = non_zero_plaq_char(group, irreps, magn_irrep)
    C = {}
    for bra in irreps_bra:
        log.info(f'>> Calculating WL mels for bra: {bra}')
        irreps_ket = product(irrep_inds, repeat=4)
        mels = {ket: wl_mel(group, irreps, ket, bra, magn_irrep, group_range)
                for ket in irreps_ket}
        if mels:
            C[bra] = mels
    return C
    # For D4, it iterates over 8^4 x 8^4 = 2^24 ~ 16mln elements
    # Not too efficient
    # Estimated time for the D4 case: 12h


class WLMatrixWorker:
    def __init__(self, group, irreps, magn_irrep):
        self.group = group
        self.irreps = irreps
        self.magn_irrep = magn_irrep
        self.group_range = non_zero_plaq_char(self.group, self.irreps, self.magn_irrep)

    def calculate_row(self, bra):
        log.info(f'>> Calculating WL mels for bra: {bra}')
        irreps_kets = product(self.irreps.mel_indices(), repeat=4)
        mels = dict()
        for ket in irreps_kets:
            mel = wl_mel(self.group, self.irreps, ket, bra, self.magn_irrep, self.group_range)
            if mel:
                mels[ket] = mel
        # mels = {
        #     ket: wl_mel(self.group, self.irreps, ket, bra, self.magn_irrep, group_range)
        #     for ket in irreps_kets
        # }
        return mels


def wl_matrix_multiproc(
        group: Group,
        irreps: Irreps,
        magn_irrep: int,
        pool_size=4
    ):
    irreps_bras = product(irreps.mel_indices(), repeat=4)
    # group_range = non_zero_plaq_char(group, irreps, magn_irrep)
    worker = WLMatrixWorker(group, irreps, magn_irrep)
    pool = Pool(pool_size)
    # TEMPORARY
    irreps_bras = list(irreps_bras)[:8]
    result = pool.map(worker.calculate_row, irreps_bras)
    return result

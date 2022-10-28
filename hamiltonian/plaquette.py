"""
Precompute the representation-theoretic coeffients of the
magnetic terms for a plaquette
"""

import logging as log
import numpy as np
import pickle
from itertools import product
from functools import reduce, cache
from operator import mul
from pathos.multiprocessing import ProcessingPool as Pool

from group import Group, Irreps
from utils.utils import sanitize
from utils.mytyping import PlaqIndex, GroupTuple


def multiply(iterable):
    """Multiply the elements of an iterable"""
    return reduce(mul, iterable, 1.0)

def all_true(iterable):
    return bool(multiply(iterable))


@cache
def prefactor(
        group: Group,
        irreps: Irreps,
        ket: PlaqIndex,
        bra: PlaqIndex
    ) -> float:
    """
    Compute the prefactor with the dimension of each irreps in `ket` and `bra`
    """
    ord_group = len(group)
    irrep_ket_dim = multiply(irreps.dim(i[0]) for i in ket)
    irrep_bra_dim = multiply(irreps.dim(i[0]) for i in bra)
    return 2 * np.sqrt(irrep_ket_dim * irrep_bra_dim) / (ord_group ** 4)


@cache
def plaq_character(
        irreps: Irreps,
        g_elems: GroupTuple,
        magn_irrep: int
    ) -> float:
    """
    Compute the character of plaquette given the magnetic irrep `magn_irrep`
    """
    g1, g2, g3, g4 = g_elems
    return np.real(irreps.chars[magn_irrep](g1 * g2 * (~g3) * (~g4)))


@cache
def plaq_mels(
        irreps: Irreps,
        g_elems: GroupTuple,
        plaq_state: PlaqIndex
    ) -> float | complex:
    """
    Computes
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
    """
    Computes one term of the sum for one matrix element of the Wilson loop
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
        log.debug(f'bra: {plaq_bra}, ket: {plaq_ket}, mel: {mel}')
        return mel
    else:
        return None
    # For D4
    #  - with full range it sums over 8^4 = 4096 elements
    #  - with non_zero_plaq_char it sums over 1024 elements
    # it is some improvement


def non_zero_plaq_char(group: Group, irreps: Irreps, magn_irrep: int):
    """
    Return a the list of states (in group repr) with
    non-zero character in the `magn_irrep` irrep
    """
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
    """
    Compute the matrix elements of the single-plaquette Wilson loop

    This function is single-threaded
    """
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
        """Computes a single row of the single-plaquette Wilson loop matrix"""
        log.info(f'>> Calculating WL mels for bra: {bra}')
        irreps_kets = product(self.irreps.mel_indices(), repeat=4)
        mels = dict()
        for ket in irreps_kets:
            mel = wl_mel(self.group, self.irreps, ket, bra, self.magn_irrep, self.group_range)
            if mel:
                mels[ket] = mel
        return mels


def wl_matrix_multiproc(
        group: Group,
        irreps: Irreps,
        magn_irrep: int,
        pool_size=4
    ):
    """
    Compute the matrix elements of the single-plaquette Wilson loop

    This function is multi-threaded, each tread computes a row of the matrix.
    The number of threads is given by `pool_size`
    """
    irreps_bras = list(product(irreps.mel_indices(), repeat=4))
    worker = WLMatrixWorker(group, irreps, magn_irrep)
    pool = Pool(pool_size)
    result = pool.map(worker.calculate_row, irreps_bras)
    return {bra: row for bra, row in zip(irreps_bras, result)}


def plaquette_links(vertices, plaquettes):
    """Returns the indices of the links the `ind`-th plaquette"""
    return [
        tuple(vertices[i][k] for k, i in enumerate(plq))
        for plq in plaquettes
    ]


class PlaquetteMel:
    """
    Class for interfacing with the matrix elements of a single plaquette Wilson loop
    """
    def __init__(self, from_dict = None, from_file = None):
        """
        Can be initialized from a dict (output of wl_matrix*) or read from a pickled file
        """
        self._mels_dict = None
        if from_dict:
            self._mels_dict = from_dict
        if from_file:
            self.load_file(from_file)
        self._rows = list(self._mels_dict.keys())
        self._irrep_confs = list({tuple(jmn[0] for jmn in row) for row in self._rows})
        self._irrep_confs.sort()

    def load_file(self, filename):
        """Load plaquette matrix elements from a file"""
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        self._mels_dict = data


    def save(self, filename):
        """Save the plaquette matrix elements to a file"""
        with open(filename, 'wb') as file:
            pickle.dump(file)


    def select_rows(self, conf):
        """
        Select the rows of the Wilson loop matrix corresponding to a
        irrep configuration on the links of the plaquette
        """
        selected_rows = {
            row: self._mels_dict[row]
            for row in self._rows
            if all_true(j == jmn[0] for j, jmn in zip(conf, row))
        }
        return selected_rows


    def select(self, bra_irreps, ket_irreps, as_list=False):
        """
        Select the matrix elements for a given bra and ket
        """
        selection = dict()
        if as_list:
            for row in self.select_rows(bra_irreps):
                for col in self._mels_dict[row]:
                    if all_true(j == jmn[0] for j, jmn in zip(ket_irreps, col)):
                        selection[(row, col)] = self._mels_dict[row][col]
        else:
            for row in self.select_rows(bra_irreps):
                selected_cols = {
                    col: self._mels_dict[row][col]
                    for col in self._mels_dict[row]
                    if all_true(j == jmn[0] for j, jmn in zip(ket_irreps, col))
                }
                if selected_cols:
                    selection[row] = selected_cols
        return selection


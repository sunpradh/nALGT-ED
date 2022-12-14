"""
Precompute the representation-theoretic coeffients of the
magnetic terms for a plaquette
"""

import logging as log
import numpy as np
import pickle

from itertools import product, chain, repeat
from functools import cache
from collections.abc import Sequence
from pathos.multiprocessing import ProcessingPool as Pool

from group import Group, Irreps
from utils.utils import sanitize, multiply, all_true, iter_irrep_mels, unpickle
from utils.mytyping import PlaqIndex, GroupTuple, IrrepIndex


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


def get_plaq_links(vertices, plaquette):
    """Returns the indices of the links the `ind`-th plaquette"""
    return tuple(vertices[i][k] for k, i in enumerate(plaquette))


class PlaquetteMels:
    """
    Class for interfacing with the matrix elements of a single plaquette Wilson loop
    """
    def __init__(
            self,
            irreps: Irreps,
            from_dict: dict | None = None,
            from_file: str | None = None
        ):
        """
        Can be initialized from a dict (output of wl_matrix*) or read from a pickled file
        """
        self._irreps = irreps
        self._mels = None
        if isinstance(from_dict, dict):
            self._mels = from_dict
        else:
            if isinstance(from_file, str):
                self.load_file(from_file)
            else:
                raise ValueError('No valid argument given')
        self._tensors = dict()


    def __len__(self):
        """Returns the number of rows"""
        return len(self._mels)


    def load_file(self, filename: str):
        """Load plaquette matrix elements from a file"""
        self._mels = unpickle(filename)


    def save(self, filename: str):
        """Save the plaquette matrix elements to a file"""
        with open(filename, 'wb') as file:
            pickle.dump(file)


    def select_rows(self, irrep_conf: Sequence[IrrepIndex]):
        """
        Select the rows of the Wilson loop matrix corresponding to a
        irrep configuration on the links of the plaquette
        """
        selected_rows = {
            row: self._mels[row]
            for row in iter_irrep_mels(irreps=self._irreps, irr_conf=irrep_conf)
            if row in self._mels
        }
        return selected_rows


    def select(
            self,
            bra_irreps: Sequence[IrrepIndex],
            ket_irreps: Sequence[IrrepIndex],
            flatten=False
        ):
        """
        Select the matrix elements for a given bra and ket
        """
        selection = dict()
        if flatten:
            for row in self.select_rows(bra_irreps):
                for col in self._mels[row]:
                    if all_true(j == jmn[0] for j, jmn in zip(ket_irreps, col)):
                        selection[(row, col)] = self._mels[row][col]
        else:
            for row in self.select_rows(bra_irreps):
                selected_cols = {
                    col: self._mels[row][col]
                    for col in iter_irrep_mels(irreps=self._irreps, irr_conf=ket_irreps)
                    if col in self._mels[row]
                }
                if selected_cols:
                    selection[row] = selected_cols
        return selection


    def _shape_from_irreps(self, irreps):
        return chain.from_iterable(
            repeat(self._irreps.dim(j), 2)
            for j in irreps
        )


    def _select_as_tensor(
            self,
            bra_irreps: Sequence[IrrepIndex],
            ket_irreps: Sequence[IrrepIndex]
        ) -> np.ndarray | None:
        selection = self.select(bra_irreps, ket_irreps, flatten=True)
        if not selection:
            return None
        shape_bra = self._shape_from_irreps(bra_irreps)
        shape_ket = self._shape_from_irreps(ket_irreps)
        C = np.zeros(tuple(chain.from_iterable((shape_bra, shape_ket))))
        for bra, ket in selection:
            index = tuple(chain.from_iterable((jmn[1], jmn[2]) for jmn in chain(bra, ket)))
            C[index] = selection[(bra, ket)]
        return C


    def tensor(
        self,
        bra_irreps: Sequence[IrrepIndex],
        ket_irreps: Sequence[IrrepIndex]
    ) -> np.ndarray | None:
        key = (bra_irreps, ket_irreps)
        if key in self._tensors:
            return self._tensors[key]
        else:
            tensor = self._select_as_tensor(bra_irreps, ket_irreps)
            self._tensors[key] = tensor
            return tensor

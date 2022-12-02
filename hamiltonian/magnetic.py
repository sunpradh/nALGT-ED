"""
Compute the magnetic Hamiltonian
"""

import numpy as np
import scipy.sparse as sparse

from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from basis.basis import Basis, State
from basis.contractions import tensor_around_plaq, contract_magnetic_elem
from hamiltonian.plaquette import PlaquetteMels, get_plaq_links
from utils.mytyping import PlaqVertices


def magn_hamiltonian_mel(
        basis: Basis,
        bra: State,
        ket: State,
        plaqs_vertices: list[PlaqVertices],
        plaq_mels: PlaquetteMels
    ) -> float | complex:
    """Compute a single matrix element of the magnetic Hamiltonian"""

    result = 0

    for p_vertices in plaqs_vertices:
        # links of the plaquette
        p_links = get_plaq_links(basis.vertices, p_vertices)
        # links outside the plaquette
        non_p_links = [link for link in range(basis.nlinks) if link not in p_links]

        bra_j_plq = tuple(bra.irreps[link] for link in p_links)
        ket_j_plq = tuple(ket.irreps[link] for link in p_links)
        bra_j_out = tuple(bra.irreps[link] for link in non_p_links)
        ket_j_out = tuple(ket.irreps[link] for link in non_p_links)

        # one-plaquette Wilson loop
        plaq_tensor = plaq_mels.tensor(bra_j_plq, ket_j_plq)

        # skip this plaquette if it has no nonzero matrix elements
        if plaq_tensor is None:
            continue

        # skip the plaquette if the irreps outside the plaquettes are not the same
        if bra_j_out != ket_j_out:
            continue

        bra_tensor = tensor_around_plaq(basis, bra, p_vertices)
        ket_tensor = tensor_around_plaq(basis, ket, p_vertices)
        result += contract_magnetic_elem(plaq_tensor, bra_tensor, ket_tensor)

    return result


class MagneticWorker:
    def __init__(
            self,
            basis: Basis,
            plaqs_vertices: list[PlaqVertices],
            plaq_mels: PlaquetteMels
        ):
        self.basis = basis
        self.plaqs_vertices = plaqs_vertices
        self.plaq_mels = plaq_mels

    def full_row(self, bra: State):
        """
        Compute an entire single row of magnetic Hamiltonian.
        Mainly used for testing
        """
        results = dict()
        for ket in self.basis.states:
            mel = magn_hamiltonian_mel(
                basis = self.basis,
                bra = bra,
                ket = ket,
                plaqs_vertices = self.plaqs_vertices,
                plaq_mels = self.plaq_mels
            )
            if mel:
                results[ket] = mel
        return results


    def partial_row(self, row_index: int):
        """
        Compute a part of a single row (only the upper triangular region)
        of the magnetic Hamiltonian
        """
        bra = self.basis.states[row_index]
        # calculate only at the right of the diagonal
        kets = self.basis.states[row_index:]
        results = dict()
        for i, ket in enumerate(kets):
            mel = magn_hamiltonian_mel(
                basis = self.basis,
                bra = bra,
                ket = ket,
                plaqs_vertices = self.plaqs_vertices,
                plaq_mels = self.plaq_mels
            )
            if mel:
                results[row_index + i] = mel
        return results


def magnetic_hamiltonian(
        basis: Basis,
        plaqs_vertices: list[PlaqVertices],
        plaq_mels: PlaquetteMels,
        progress_bar = False
    ) -> sparse.dok_matrix:
    """Compute the entire magnetic Hamiltonian"""
    worker = MagneticWorker(basis, plaqs_vertices, plaq_mels)
    n_states = len(basis.states)
    H = sparse.dok_matrix((n_states, n_states))
    iterator = tqdm(range(n_states)) if progress_bar else range(n_states)
    for row_ind in iterator:
        row = worker.partial_row(row_ind)
        for col_ind, elem in row.items():
            H[row_ind, col_ind] = elem
            H[col_ind, row_ind] = np.conj(elem)
    return H


def magnetic_hamiltonian_mp(
        basis: Basis,
        plaqs_vertices: list[PlaqVertices],
        plaq_mels: PlaquetteMels,
        pool_size: int = 4
    ) -> sparse.dok_matrix:
    """
    Compute the entire magnetic Hamiltonian (Multiprocessing version).
    Does not work very well
    """
    worker = MagneticWorker(basis, plaqs_vertices, plaq_mels)
    pool = Pool(pool_size)
    n_states = len(basis.states)
    lst = list(range(n_states))
    pool_result = pool.map(worker.partial_row, lst)
    H = sparse.dok_matrix((n_states, n_states))
    for row_ind, row in zip(lst, pool_result):
        for col_ind, elem in row.items():
            H[row_ind, col_ind] = elem
            H[col_ind, row_ind] = np.conj(elem)
    return H

"""
Compute the magnetic Hamiltonian
"""

import logging as log
import numpy as np
import scipy.sparse as sparse
from collections.abc import Sequence
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from basis.basis import Basis, State
from hamiltonian.plaquette import PlaquetteMels, plaquette_links
from utils.mytyping import MelIndex, VertexLinks, PlaqVertices
from utils.utils import iter_irrep_mels


def subindices_vertex(
        vertex: VertexLinks,
        jmns: Sequence[MelIndex]
    ) -> tuple[int, int, int, int]:
    """Return the indices (m1, m2, n3, n4) for a vertex v"""
    return (
        jmns[vertex[0]][1],
        jmns[vertex[1]][1],
        jmns[vertex[2]][2],
        jmns[vertex[3]][2]
    )


def compute_psi(
        psi: np.ndarray,
        vertex: VertexLinks,
        jmns: Sequence[MelIndex]
    ) -> float:
    m1, m2, n3, n4 = subindices_vertex(vertex, jmns)
    return psi[m1][m2][n3][n4]


def compute_psi_plaq(
        basis: Basis,
        state: State,
        plaquette: PlaqVertices,
        jmns: Sequence[MelIndex]
    ) -> float:
    result = 1
    for v_i in plaquette:
        vertex = basis.vertices[v_i]
        psi = basis(state)[v_i]
        result = result * compute_psi(psi, vertex, jmns)
    return result


def magnetic_hamiltonian_mel(
        basis: Basis,
        bra: State,
        ket: State,
        plaqs_vertices: list[PlaqVertices],
        plaq_mels: PlaquetteMels
    ):
    # plaquette links
    plaqs_links = plaquette_links(basis.vertices, plaqs_vertices)

    # We are gonna use a brute force method:
    # Cycle over all the possible subindices (m_i, n_i) given (j_1, ..., j_nlinks)
    # both for the bra and ket states

    # Cycle over all the possible plaquettes
    result = 0
    for p_vertices, p_links in zip(plaqs_vertices, plaqs_links):
        # links outside the plaquette
        non_p_links = [link for link in range(basis.nlinks) if link not in p_links]

        bra_j_plq = tuple(bra.irreps[link] for link in p_links)
        ket_j_plq = tuple(ket.irreps[link] for link in p_links)
        bra_j_out = tuple(bra.irreps[link] for link in non_p_links)
        ket_j_out = tuple(ket.irreps[link] for link in non_p_links)

        # one-plaquette Wilson loop
        WL = plaq_mels.select(bra_j_plq, ket_j_plq, flatten=False)

        # skip this plaquette if it has no nonzero matrix elements
        if not WL:
            continue

        # skip the plaquette if the irreps outside the plaquettes are not the same
        if bra_j_out != ket_j_out:
            continue

        # indices (m_i, n_i) for the bra state
        for bra_jmn in iter_irrep_mels(basis.irreps, bra.irreps):

            bra_jmn_plq = tuple(bra_jmn[link] for link in p_links)
            bra_jmn_out = tuple(bra_jmn[link] for link in non_p_links)

            # check if there are non-zero matrix elements of WL to sum over
            if bra_jmn_plq not in WL:
                continue

            # indices (m_i, n_i) for the ket state
            for ket_jmn in iter_irrep_mels(basis.irreps, ket.irreps):
                ket_jmn_plq = tuple(ket_jmn[link] for link in p_links)
                ket_jmn_out = tuple(ket_jmn[link] for link in non_p_links)

                # check if the indices outside the plaquette are the same
                if bra_jmn_out != ket_jmn_out:
                    continue

                # single plaquette wilson loop matrix element
                if ket_jmn_plq not in WL[bra_jmn_plq]:
                    continue

                C = WL[bra_jmn_plq][ket_jmn_plq]

                # product of the gauge inv. coeff. for the bra and ket state
                psi_bra = np.conj(compute_psi_plaq(basis, bra, p_vertices, bra_jmn))
                psi_ket = compute_psi_plaq(basis, ket, p_vertices, ket_jmn)

                # loggin information
                log.info("computing something:")
                log.info(f"\t bra_jmn: {bra_jmn}")
                log.info(f"\t ket_jmn: {ket_jmn}")
                log.info(f"\t psi_bra * C * psi_ket: {psi_bra * C * psi_ket}")

                # add to the sum
                result = result + (psi_bra * C * psi_ket)
    return result


def magnetic_hamiltonian_row(
        basis: Basis,
        bra: State,
        plaquettes: list[PlaqVertices],
        plaquette_mels: PlaquetteMels,
        progress_bar=False
    ):
    results = dict()
    iterator = tqdm(basis.states) if progress_bar else basis.states
    for ket in iterator:
        c = magnetic_hamiltonian_mel(basis, bra, ket, plaquettes, plaquette_mels)
        if c:
            results[ket] = c
    return results


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

    def calculate_row(self, row: int):
        bra = self.basis.states[row]
        # calculate only at the right of the diagonal
        kets = self.basis.states[row:]
        results = dict()
        print(f"magnetic worker for row = {row}")
        for i, ket in enumerate(kets):
            mel = magnetic_hamiltonian_mel(
                basis = self.basis,
                bra = bra,
                ket = ket,
                plaqs_vertices = self.plaqs_vertices,
                plaq_mels = self.plaq_mels
            )
            if mel:
                results[row + i] = mel
        return results

def magnetic_hamiltonian(
        basis: Basis,
        plaqs_vertices: list[PlaqVertices],
        plaq_mels: PlaquetteMels,
        pool_size: int = 4
    ) -> sparse.dok_matrix:
    worker = MagneticWorker(basis, plaqs_vertices, plaq_mels)
    pool = Pool(pool_size)
    n_states = len(basis.states)
    lst = list(range(3))
    pool_result = pool.map(worker.calculate_row, lst)
    H = sparse.dok_matrix((n_states, n_states))
    for row_ind, row in zip(lst, pool_result):
        for col_ind, elem in row.items():
            H[row_ind, col_ind] = elem
            H[col_ind, row_ind] = np.conj(elem)
    return H

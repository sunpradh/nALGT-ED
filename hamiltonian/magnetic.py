"""
Compute the magnetic Hamiltonian
"""

import numpy as np
from collections.abc import Sequence
from tqdm import tqdm

from basis.basis import Basis, State
from hamiltonian.plaquette import PlaquetteMel, plaquette_links
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
        plaquettes: list[PlaqVertices],
        plaquette_mels: PlaquetteMel
    ):
    # plaquette links
    plaqs_links = plaquette_links(basis.vertices, plaquettes)

    # We are gonna use a brute force method:
    # Cycle over all the possible subindices (m_i, n_i) given (j_1, ..., j_nlinks)
    # both for the bra and ket states

    # Cycle over all the possible plaquettes
    result = 0
    for p_vertices, p_links in zip(plaquettes, plaqs_links):
        # links outside the plaquette
        non_p_links = [link for link in range(basis.nlinks) if link not in p_links]

        bra_j_plq = tuple(bra.irreps[link] for link in p_links)
        ket_j_plq = tuple(ket.irreps[link] for link in p_links)
        bra_j_out = tuple(bra.irreps[link] for link in non_p_links)
        ket_j_out = tuple(ket.irreps[link] for link in non_p_links)

        # one-plaquette Wilson loop
        WL = plaquette_mels.select(bra_j_plq, ket_j_plq, flatten=False)

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

                # print("computing something")
                C = WL[bra_jmn_plq][ket_jmn_plq]

                # product of the gauge inv. coeff. for the bra and ket state
                psi_bra = np.conj(compute_psi_plaq(basis, bra, p_vertices, bra_jmn))
                psi_ket = compute_psi_plaq(basis, ket, p_vertices, ket_jmn)

                # add to the sum
                result = result + (psi_bra * C * psi_ket)
    return result


def magnetic_hamiltonian_row(
        basis: Basis,
        bra: State,
        plaquettes: list[PlaqVertices],
        plaquette_mels: PlaquetteMel,
        progress_bar=False
    ):
    results = dict()
    iterator = tqdm(basis.states) if progress_bar else basis.states
    for ket in iterator:
        c = magnetic_hamiltonian_mel(basis, bra, ket, plaquettes, plaquette_mels)
        if c:
            results[ket] = c
    return results

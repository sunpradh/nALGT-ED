"""
Compute the Hamiltonian
"""
import numpy as np
from itertools import product
from collections.abc import Callable, Sequence

from basis.basis import Basis, StateLabel
from group import Group_elem, Irreps
from plaquette import PlaquetteMel, plaquette_links
from mytyping import MelIndex, VertexLinks, PlaqVertices


def electric_single_link_fn(
        generating_set: list[Group_elem],
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


def indices_list(
        basis: Basis,
        state: StateLabel
    ) -> list[Sequence[MelIndex]]:
    """
    Return a list of all the possible subindices (j_i, m_i, n_i) for each link
    given a state with a irrep conf (j_1, ..., j_nlinks)
    """
    # possible subindices for each link
    return [
        list(
            product([j], range(basis.irreps.dim(j)), range(basis.irreps.dim(j)))
        )
        for j in state.irreps
    ]


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


def compute_psi_plq(
        basis: Basis,
        state: StateLabel,
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
        bra: StateLabel,
        ket: StateLabel,
        plaquettes: list[PlaqVertices],
        plaquette_mels: PlaquetteMel
    ):
    # plaquette links
    plaqs_links = plaquette_links(basis.vertices, plaquettes)

    # subindices (m_i, n_i) for the bra and ket states
    bra_indices_list = indices_list(basis, bra)
    ket_indices_list = indices_list(basis, ket)

    # We are gonna use a brute force method:
    # Cycle over all the possible subindices (m_i, n_i) given (j_1, ..., j_nlinks)
    # both for the bra and ket states

    # Cycle over all the possible plaquettes
    result = 0
    for p_vertices, p_links in zip(plaquettes, plaqs_links):
        bra_plaq_js = tuple(bra.irreps[link] for link in p_links)
        ket_plaq_js = tuple(ket.irreps[link] for link in p_links)
        # one-plaquette Wilson loop
        WL = plaquette_mels.select(bra_plaq_js, ket_plaq_js, as_list=False)

        # skip this plaquette if it has no nonzero matrix elements
        if not WL:
            continue

        non_p_links = [link for link in range(basis.nlinks) if link not in p_links]

        for bra_jmn in product(*bra_indices_list):
            bra_jmn_plq = tuple(bra_jmn[link] for link in p_links)
            bra_jmn_out = tuple(bra_jmn[link] for link in non_p_links)

            # check if there are non-zero matrix elements of WL to sum over
            if bra_jmn_plq not in WL:
                continue

            for ket_jmn in product(*ket_indices_list):
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
                psi_bra = np.conj(compute_psi_plq(basis, bra, p_vertices, bra_jmn))
                psi_ket = compute_psi_plq(basis, ket, p_vertices, ket_jmn)

                # add to the sum
                result = result + (psi_bra * C * psi_ket)
    return result

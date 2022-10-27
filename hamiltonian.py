"""
Compute the Hamiltonian
"""
import numpy as np
from itertools import product
from basis.basis import Basis, StateLabel
from group import Group_elem, Irreps
from typing import Callable
from plaquette import PlaquetteMel, plaquette_links

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


def subindices_list(basis: Basis, state: StateLabel):
    """
    Return a list of all the possible subindices (m, n) for each link
    given a state with a irrep conf (j_1, ..., j_nlinks)
    """
    # possible subindices for each link
    subindices = [
        list(product(range(basis.irreps.dim(j)), repeat=2))
        for j in state.irreps
    ]
    # list of all the possible combination of subindices
    return list(product(*subindices))


def magnetic_hamiltonian_mel(
        basis: Basis,
        bra: StateLabel,
        ket: StateLabel,
        plaquettes: list[tuple[int]],
        wilsonloop_data: PlaquetteMel
    ):
    # We are gonna use a brute force method

    # plaquette links
    plq_links = plaquette_links(basis.vertices, plaquettes)

    # subindices (m_i, n_i) for the bra and ket states
    bra_iter = subindices_list(basis, bra)
    ket_iter = subindices_list(basis, ket)

    def jmn_vertex(vertex: list[int], jmns: tuple[tuple[int, int, int], ...]):
        return (
            jmns[vertex[0]][1],
            jmns[vertex[1]][1],
            jmns[vertex[2]][2],
            jmns[vertex[3]][2]
        )

    def compute_psi(psi: np.ndarray, vertex: list[int], jmns: tuple[tuple[int, int, int], ...]):
        m1, m2, n3, n4 = jmn_vertex(vertex, jmns)
        return psi[m1][m2][n3][n4]

    def compute_psi_plq(basis: Basis, state: StateLabel, plaquette: list[tuple[int, ...]], jmns):
        result = 1
        for v_i in plaquette:
            vertex = basis.vertices[v_i]
            psi = basis(state)[v_i]
            result = result * compute_psi(psi, vertex, jmns)
        return result


    # Cycle over all the bra subindices given (j_1, ..., j_nlinks)
    result = 0
    for vertices, links in zip(plaquettes, plq_links):
        # Cycle over all the possible plaquettes
        bjs = tuple(bra.irreps[link] for link in links)
        kjs = tuple(ket.irreps[link] for link in links)
        WL = wilsonloop_data.select(bjs, kjs, as_list=False)
        if not WL:
            continue
        for bra_mns in bra_iter:
            # jmn indices for the bra state
            bra_jmn = tuple((j, *mn) for j, mn in zip(bra.irreps, bra_mns) )
            bra_jmn_plq = tuple(jmn for link, jmn in enumerate(bra_jmn) if link in links)
            bra_jmn_out = tuple(jmn for link, jmn in enumerate(bra_jmn) if link not in links)

            # check if there are matrix elements of C with the given bra state
            if bra_jmn_plq not in WL:
                continue

            for ket_mns in ket_iter:
                # jmn indicies for the ket state
                ket_jmn = tuple( (j, *mn) for j, mn in zip(ket.irreps, ket_mns) )
                ket_jmn_plq = tuple(jmn for link, jmn in enumerate(ket_jmn) if link in links)
                ket_jmn_out = tuple(jmn for link, jmn in enumerate(ket_jmn) if link not in links)

                # check if the indices outside the plaquette are the same
                if bra_jmn_out != ket_jmn_out:
                    continue

                # single plaquette wilson loop matrix element
                if ket_jmn_plq not in WL[bra_jmn_plq]:
                    continue
                C = WL[bra_jmn_plq][ket_jmn_plq]

                # product of the gauge inv. coeff. for the bra and ket state
                psi_bra = np.conj(compute_psi_plq(basis, bra, vertices, bra_jmn))
                psi_ket = compute_psi_plq(basis, ket, vertices, ket_jmn)

                # add to the sum
                result = result + (psi_bra * C * psi_ket)
    return result

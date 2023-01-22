# Set number of threads to use (before loading numpy and scipy)
import os
os.environ['OMP_NUM_THREADS'] = '48'

import numpy as np
from scipy.sparse.linalg import eigsh

from group import DihGroup, DihIrreps
from basis import Basis
from hamiltonian import elec_hamiltonian
from tests.lattice_2x2 import vertices, nlinks
from utils.utils import unpickle

group = DihGroup(4)
irreps = DihIrreps(group.N)

print(f'> Group: {group}')
print(f'> Irreps: {irreps}')
print("> Computing physical Hilbert space")
basis = Basis(group, irreps, vertices, nlinks)
print(f"\ttotal number of states: {len(basis.states)}")

#------------------------------------------------------------
# Helpful methods
#------------------------------------------------------------

def expt_value(matrix, vector):
    """Compute expectation value given vector and matrix"""
    return vector @ matrix @ vector


def eigstates(coupling, elec_hamil, magn_hamil, n_eigs):
    """Compute eigenvalues and eigenvectors for a given coupling"""
    print(f'\tλ = {coupling:.5f}\t', end='')
    H = (1 - coupling) * elec_hamil - coupling * magn_hamil
    energies, vecs = eigsh(H, k=n_eigs, which='SA')
    eigvecs = [vec.ravel() for vec in vecs.T]
    return energies, eigvecs


def eigstates_over_range(coupling_range, elec_hamil, magn_hamil, n_eigs):
    """Compute eigenvalues and eigenvectors over a range of couplings"""
    n_couplings = len(coupling_range)
    results = dict(
        energies = np.zeros((n_couplings, n_eigs)),
        expt_elec = np.zeros(n_couplings),
        expt_magn = np.zeros(n_couplings),
        ground_states = np.zeros((n_couplings, elec_hamil.shape[0]))
    )
    print('\n>> Computing eigenvalues and eigenvectors\n')
    for n, coupling in enumerate(coupling_range):
        energies, eigvecs = eigstates(coupling, elec_hamil, magn_hamil, n_eigs)
        gs = eigvecs[0]
        results['energies'][n, :] = energies
        print(f'E0 = {results["energies"][n, 0]:.5f}\t', end='')
        results['ground_states'][n] = gs
        results['expt_elec'][n] = expt_value(elec_hamil, gs)
        results['expt_magn'][n] = expt_value(magn_hamil, gs)
        print(f'<H_E> = {results["expt_elec"][n]:.5f} \t', end='')
        print(f'<H_B> = {results["expt_magn"][n]:.5f}')
    print()
    return results


def save_results(results: dict, name: str):
    """Save results in a .npz format"""
    print(f'Saving for "{name}"')
    np.savez_compressed(name, **results)


def load_elec_hamiltonian(basis, irreps, gen_set):
    """Load the electric hamiltonian for the given generating set"""
    print('> Computing Electric Hamiltonian')
    elec_hamil = elec_hamiltonian(basis, gen_set, irreps, progress_bar=True).tocsc()
    print('> loaded')
    print(f'\t{repr(elec_hamil)}')
    return elec_hamil


def compute(gen_set, couplings, n_eigs, name):
    """Compute and then save"""
    HE = load_elec_hamiltonian(basis, irreps, gen_set)
    results = eigstates_over_range(couplings, HE, HB, n_eigs)
    results['couplings'] = couplings
    save_results(results, name)
    print()


#------------------------------------------------------------
# Useful objects
#------------------------------------------------------------

## Common objects for the three classes of electric Hamiltonians
# couplings = np.linspace(0, 1, 3) # testing case

# Load electric and magnetic hamiltonian as dok (dict of keys) sparse matrices
# and convert them to compressed sparse column

# Magnetic Hamiltonian
print('> Loading Magnetic Hamiltonian')
HB_dok = unpickle("pickled/magn_hamiltonian_D4_2x2.old.pkl") # old but correct Hamiltonian
HB = HB_dok.todense()
print('\tloaded')
print(f'\t{repr(HB)}')
print()

# number of energy levels
# num_eigs = 24

# Group generators
r = group.r
s = group.s

# printing options
np.set_printoptions(precision=6, linewidth=120)


#------------------------------------------------------------
# Main computation
#------------------------------------------------------------

print('----------------------------------------')
print(' Non-relativistic case')
print('----------------------------------------')
gen_set_NR = {r, ~r, s, r*r*s}
couplings_NR = np.concatenate((
                  np.linspace(0, 0.6, 61)[:-1],
                  np.linspace(0.6, 0.8, 101),
                  np.linspace(0.8, 1, 21)[1:]
              ))
compute(
    gen_set=gen_set_NR,
    couplings=couplings_NR,
    n_eigs=10,
    name='results_NR'
)


print('----------------------------------------')
print(' Relativistic case')
print('----------------------------------------')
gen_set_R = {r, ~r, s, r*s, r*r*s, r*r*r*s}
couplings_R = np.concatenate((
                  np.linspace(0, 0.65, 66)[:-1],
                  np.linspace(0.65, 0.85, 101),
                  np.linspace(0.85, 1, 16)[1:]
              ))
compute(
    gen_set=gen_set_R,
    couplings=couplings_R,
    n_eigs=10,
    name='results_R'
)


print('----------------------------------------')
print(' Degenerate case')
print('----------------------------------------')
gen_set_D = {r, r*r, r*r*r}
couplings_D = np.concatenate((
                  np.linspace(0, 0.5, 51)[:-1],
                  np.linspace(0.5, 0.8, 151),
                  np.linspace(0.8, 1, 21)[1:]
              ))
compute(
    gen_set=gen_set_D,
    couplings=couplings_D,
    n_eigs=40,
    name='results_D'
)

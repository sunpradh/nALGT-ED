# Set number of threads to use (before loading numpy and scipy)
import os
os.environ['OMP_NUM_THREADS'] = '12'

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

generating_set = {group((1,0)), group((3,0)), group((0, 1)), group((2, 1))} # {r, r3, s, r2s}

# load electric and magnetic hamiltonian as dok (dict of keys) sparse matrices
# and convert them to dense matrices
print('> Computing Electric Hamiltonian')
HE_dok = elec_hamiltonian(basis, generating_set, irreps, progress_bar=True)
HE = HE_dok.todense()
print('\tloaded')
print(f'\t{repr(HE)}')
print('> Loading Magnetic Hamiltonian')
# Old hamiltonian
HB_dok = unpickle("pickled/magn_hamiltonian_D4_2x2.old.pkl")
HB = HB_dok.todense()
print('\tloaded')
print(f'\t{repr(HB)}')
print()

# couplings
couplings = np.linspace(0, 1, 101)
# couplings = [0, 0.8, 1]

# number of energy levels
n_eigs = 24
# store energy levels
energies = np.zeros((len(couplings), n_eigs))
expt_HE = np.zeros(len(couplings))
expt_HB = np.zeros(len(couplings))
ground_states = np.zeros(len(couplings), HE.shape[0])

def expt_value(matrix, vector):
    return vector @ matrix @ vector

np.set_printoptions(precision=6, linewidth=120)
print('> Computing spectrum')

for n, lamb in enumerate(couplings):
    print(f'\tλ = {lamb:.5f}\t', end='')
    H = (1 - lamb) * HE + lamb * HB
    E, V = eigsh(H, k=n_eigs, which='SA')

    print(f'E0 = {E[0]:.4f}\t', end='')
    eigvecs = [vec.ravel() for vec in V.T]
    energies[n,:] = E
    ground_states[n, :] = eigvecs[0]

    expt_HE[n] = expt_value(HE, eigvecs[0])
    expt_HB[n] = expt_value(HB, eigvecs[0])
    print(f'<H_E> = {expt_HE[n]:.5f}  \t<H_B> = {expt_HB[n]:.5f}')

np.save('energies', energies)
np.save('expt_HE', expt_HE)
np.save('expt_HB', expt_HB)
np.save('ground_states', ground_states)

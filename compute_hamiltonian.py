from group import DihGroup, DihIrreps
from basis.basis import Basis
from hamiltonian.plaquette import PlaquetteMels
from hamiltonian.magnetic import magnetic_hamiltonian
from tests.lattice_2x2 import vertices, plaqs_vertices, nlinks
import pickle

group = DihGroup(4)
irreps = DihIrreps(group.N)
filename = 'magn_hamilt_D4.pkl'

print("> Computing physical Hilbert space")
basis = Basis(group, irreps, vertices, nlinks)
print(f"\ttotal number of states: {len(basis.states)}\n")

print('> Loading single plaquette matrix elements')
plaq_mels = PlaquetteMels(irreps=irreps, from_file="pickled/plaquette_data_D4.pkl")
print('> Plaquette loaded')
print(f'\t#rows: {len(plaq_mels)}\n')

H_B = magnetic_hamiltonian(basis, plaqs_vertices, plaq_mels)

with open(filename, 'wb') as file:
    pickle.dump(H_B, file)

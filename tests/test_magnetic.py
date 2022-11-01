import sys
if '..' not in sys.path:
    sys.path.append('..')

import logging as log
# log.basicConfig(level=log.INFO, format=" >> %(message)s")

from group import Dih_group, Dih_Irreps
from basis.basis import Basis, State
from hamiltonian.plaquette import PlaquetteMel
from hamiltonian.magnetic import magnetic_hamiltonian_mel
from tests.lattice_2x2 import vertices, plaq_vertices, nlinks

group = Dih_group(4)
irreps = Dih_Irreps(group.N)

print(f'> Group: {group}')
print(f'> Irreps: {irreps}')

print("> Computing physical Hilbert space")
basis = Basis(group, irreps, vertices, nlinks)
print(f"\ttotal number of states: {len(basis.states)}\n")

print('> Loading single plaquette matrix elements')
plaq_mels = PlaquetteMel(irreps=irreps, from_file="../pickled/plaquette_data_D4.pkl")
print('> Plaquette loaded')
print(f'\t#rows: {len(plaq_mels)}\n')

bra0 = basis.states[0]
ket0 = State((0, 0, 0, 0, 4, 4, 4, 4), (0, 0, 0, 0))
print('> Single matrix element:')
print(f"bra0: {bra0}")
print(f"ket0: {ket0}")
c = magnetic_hamiltonian_mel(basis, bra0, ket0, plaq_vertices, plaq_mels)
print(f"<bra0|H_B|ket0> = {c}")

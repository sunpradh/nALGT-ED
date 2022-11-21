import sys
if '..' not in sys.path:
    sys.path.append('..')

# import logging as log
# log.basicConfig(level=log.INFO, format=" >> %(message)s")

from group import DihGroup, DihIrreps
from basis.basis import Basis, State
from hamiltonian.plaquette import PlaquetteMels
from hamiltonian.magnetic import magnetic_hamiltonian_mel, magnetic_hamiltonian_row
from tests.lattice_2x2 import vertices, plaq_vertices, nlinks

group = DihGroup(4)
irreps = DihIrreps(group.N)

print(f'> Group: {group}')
print(f'> Irreps: {irreps}')

print("> Computing physical Hilbert space")
basis = Basis(group, irreps, vertices, nlinks)
print(f"\ttotal number of states: {len(basis.states)}\n")

print('> Loading single plaquette matrix elements')
plaq_mels = PlaquetteMels(irreps=irreps, from_file="../pickled/plaquette_data_D4.pkl")
print('> Plaquette loaded\n')

# Test on single states
bra0 = basis.states[0]
kets0 = [
    State((4, 4, 4, 4, 0, 0, 0, 0), (0, 0, 0, 0)),
    State((0, 0, 0, 0, 4, 4, 4, 4), (0, 0, 0, 0)),
    State((0, 4, 0, 4, 4, 4, 0, 0), (0, 0, 0, 0)),
    State((4, 0, 4, 0, 0, 0, 4, 4), (0, 0, 0, 0))
]

for ket0 in kets0:
    print('> Single matrix element:')
    print(f"bra0: {bra0}")
    print(f"ket0: {ket0}")
    c = magnetic_hamiltonian_mel(basis, bra0, ket0, plaq_vertices, plaq_mels)
    print(f"<bra0|H_B|ket0> = {c}")
    print()
print()

# Test on a full row for some candidate bra
bras = [
    State((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0)),
    State((1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0)),
    State((2, 2, 2, 2, 2, 2, 2, 2), (0, 0, 0, 0)),
    State((3, 3, 3, 3, 3, 3, 3, 3), (0, 0, 0, 0))
]

for bra in bras:
    print(f'> Calcuting row for bra = {bra}')
    result_row = magnetic_hamiltonian_row(basis, bra, plaq_vertices, plaq_mels, progress_bar=True)
    print('> Resulting states:')
    for state, val in result_row.items():
        print(f'\t{state} -> {val}')
    print()

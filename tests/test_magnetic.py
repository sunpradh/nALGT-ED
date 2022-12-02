import sys
if '..' not in sys.path:
    sys.path.append('..')

from group import DihGroup, DihIrreps
from basis.basis import Basis, State
from hamiltonian.plaquette import PlaquetteMels
from hamiltonian.magnetic import magn_hamiltonian_mel, MagneticWorker
from tests.lattice_2x2 import vertices, plaqs_vertices, nlinks

group = DihGroup(4)
irreps = DihIrreps(group.N)

print(f'> Group: {group}')
print(f'> Irreps: {irreps}')

print('> Computing physical Hilbert space')
basis = Basis(group, irreps, vertices, nlinks)
print(f'\ttotal number of states: {len(basis.states)}\n')

print('> Loading single plaquette matrix elements')
plaq_mels = PlaquetteMels(irreps=irreps, from_file="../pickled/plaquette_data_D4.pkl")
print('\tPlaquette loaded')
print(f'\t#rows: {len(plaq_mels)}')

# Testing on single matrix elements
bra0 = State((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0))
kets0 = [
    State((4, 4, 4, 4, 0, 0, 0, 0), (0, 0, 0, 0)),
    State((0, 0, 0, 0, 4, 4, 4, 4), (0, 0, 0, 0)),
    State((0, 4, 0, 4, 4, 4, 0, 0), (0, 0, 0, 0)),
    State((4, 0, 4, 0, 0, 0, 4, 4), (0, 0, 0, 0))
]

print('> Computing single matrix elements')

for ket0 in kets0:
    print(f'\tbra0: {bra0}')
    print(f'\tket0: {ket0}')
    mel = magn_hamiltonian_mel(basis, bra0, ket0, plaqs_vertices, plaq_mels)
    print(f"\t<bra0|H_B|ket0> = {mel}")
    print()
print()

# Test on a full row for some candidate bra
bras = [
    State((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0)),
    State((1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0)),
    State((2, 2, 2, 2, 2, 2, 2, 2), (0, 0, 0, 0)),
    State((3, 3, 3, 3, 3, 3, 3, 3), (0, 0, 0, 0)),
    State((0, 0, 0, 0, 4, 4, 4, 4), (0, 0, 0, 0))
]

magn_worker = MagneticWorker(
                    basis=basis,
                    plaqs_vertices=plaqs_vertices,
                    plaq_mels=plaq_mels
                )


print('> Computing single rows')
for bra in bras:
    print(f'\t> bra = {bra}')
    result_row = magn_worker.full_row(bra)
    print(f'> #states: {len(result_row)}')
    for state, val in result_row.items():
        print(f'\t{state} -> {val}')
    print()

import sys
sys.path.append('..')

from group import Dih_group, Dih_Irreps
from basis.basis import Basis

group = Dih_group(4)
irreps = Dih_Irreps(group.N)

# 2x2 periodic lattice
vertices = [
    (0, 3, 4, 7),
    (4, 1, 0, 6),
    (5, 6, 2, 1),
    (2, 7, 5, 3)
]

nlinks = 8

print(f'> Group: {group}')
print(f'> Irreps: {group}')
print("> Computing physical Hilbert space")
basis = Basis(group, irreps, vertices, nlinks)
print(f'> Total number of irreps confs: {len(basis._basis)}')
print(f'> Total number of physical states: {len(basis.states)}')

expected_num_states = sum(
    (len(group)/len(C))**len(vertices) for C in group.conj_classes()
)
print(f'> Total number of expectect physical states: {expected_num_states}')

print(f'> Equal? {len(basis.states) == expected_num_states}')

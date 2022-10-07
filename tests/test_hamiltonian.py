import sys
sys.path.append('..')

from group import Dih_group, Dih_Irreps
from basis.basis import Basis
from hamiltonian import electric_hamiltonian

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
print(f"\ttotal number of states: {len(basis.states)}")

generating_set = {group((1,0)), group((3,0)), group((0, 1)), group((2, 1))}
inv_gen_set = {~g for g in generating_set}
if generating_set != inv_gen_set:
    raise ValueError("the generating set is not valid")
else:
    print(f"> Valid generating set: {generating_set}")

print("> Computing electric Hamiltonian")
H_E = electric_hamiltonian(basis, generating_set, irreps)
print("> Done")

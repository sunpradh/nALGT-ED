import sys
if '..' not in sys.path:
    sys.path.append('..')

from group import DihGroup, DihIrreps
from basis.basis import Basis
from hamiltonian.electric import elec_hamiltonian
from lattice_2x2 import vertices, nlinks

group = DihGroup(4)
irreps = DihIrreps(group.N)

print(f'> Group: {group}')
print(f'> Irreps: {irreps}')
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
H_E = elec_hamiltonian(basis, generating_set, irreps, progress_bar=True)
print("> Done")

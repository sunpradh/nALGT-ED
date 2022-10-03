import sys
sys.path.append('..')

import logging as log

from group import Dih_group, Dih_Irreps
# from group import Q8, Q8_Irreps
from basis.basis import physical_basis

log.basicConfig(level=log.INFO)


group = Dih_group(4)
irreps = Dih_Irreps(group.N)
# group = Q8()
# irreps = Q8_Irreps()

# 2x2 periodic lattice
vertices = [
    (0, 3, 4, 7),
    (4, 1, 0, 6),
    (5, 6, 2, 1),
    (2, 7, 5, 3)
]

nlinks = 8

basis = physical_basis(group, irreps, vertices, nlinks)
print(f'Total number of irreps confs: {len(basis)}')

import pickle
with open(f"../pickled/basis_{group}.pkl", "wb") as file:
    print(f"> Wrinting to {file.name}")
    pickle.dump(basis, file)

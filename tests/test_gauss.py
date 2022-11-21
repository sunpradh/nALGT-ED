import sys
if '..' not in sys.path:
    sys.path.append('..')

import numpy as np

from group import DihGroup, DihIrreps
group = DihGroup(4)
irreps = DihIrreps(group.N)

from basis.gauss import gauss_operator
from basis.invariant import irrep_conf

confs = [
    (4, 4, 0, 0),
    (0, 4, 4, 0),
    (0, 0, 4, 4),
    (4, 0, 0, 4)
]

Gr = np.array([
                  [0, 0,  0,  1],
                  [0, 0,  -1, 0],
                  [0, -1, 0,  0],
                  [1, 0,  0,  0],
              ])
Gs = np.array([
                  [1, 0,  0,  0],
                  [0, -1, 0,  0],
                  [0, 0,  -1, 0],
                  [0, 0,  0,  1],
              ])

for g, G_correct in zip([group.r, group.s], [Gr, Gs]):
    for conf in confs:
        G = gauss_operator(g, irrep_conf(conf, irreps), sanitized=True)
        print(f"> Gauss operator for {repr(g)} and j={conf}")
        print(G)
        print(f"\t>> correct? {np.all(G == G_correct)}")
        print()

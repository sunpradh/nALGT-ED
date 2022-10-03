import sys
sys.path.append('..')

import logging as log
import pickle

from group import Dih_group, Dih_Irreps
from plaquette import wl_matrix_multiproc

log.basicConfig(level=log.INFO)

group = Dih_group(4)
irreps = Dih_Irreps(group.N)

plaq_matrix = wl_matrix_multiproc(group, irreps, 4, pool_size=8)

with open("../pickled/plaquette_data.pkl", "wb") as file:
    pickle.dump(plaq_matrix, file)

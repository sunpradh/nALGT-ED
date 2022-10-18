import sys
sys.path.append('..')

import logging as log

from group import Dih_group, Dih_Irreps
from plaquette import Plaquette, plaq_links

log.basicConfig(level=log.INFO)

group = Dih_group(4)
irreps = Dih_Irreps(group.N)

# plaq_matrix = wl_matrix_multiproc(group, irreps, 4, pool_size=8)
# plaq = Plaquette(from_dict=plaq_matrix)

plaq = Plaquette(from_file="../pickled/plaquette_data_D4.pkl")
print('> Plaquette loaded')
print(f'\t> #rows: {len(plaq._rows)}')
print(f'\t> #irreps confs: {len(plaq._irrep_confs)}')
print()

conf1 = (0, 0, 0, 0)
print(f'> Selected irrep conf: {conf1}')
print(f'\t> #confs: {len(plaq.select_rows(conf1))}')
print(f'\t> expected: {1}')

conf2 = (4, 4, 4, 4)
print(f'> Selected irrep conf: {conf2}')
print(f'\t> #confs: {len(plaq.select_rows(conf2))}')
print(f'\t> expected: {4**4}')

conf3 = (4, 0, 0, 4)
print(f'> Selected irrep conf: {conf3}')
print(f'\t> #confs: {len(plaq.select_rows(conf3))}')
print(f'\t> expected: {4**2}')


# 2x2 periodic lattice
# indices of the links
vertices = [
    (0, 3, 4, 7),
    (4, 1, 0, 6),
    (5, 6, 2, 1),
    (2, 7, 5, 3)
]

# indices of the vertices for each plaquettes
# counterclockwise, from the bottom left
plaquettes = [
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 1, 0)
]

# expected plaquettes links
plaquette_links = [
    (0, 1, 2, 3),
    (4, 3, 5, 1),
    (5, 7, 4, 6),
    (2, 6, 0, 7)
]

# Testing function plaq_links
print("\n\nPlaquettes links:")
for k in range(len(plaquettes)):
    links = plaq_links(vertices, plaquettes, k)
    print(f"> Plaq n. {k}:")
    print(f"\t> computed: {links}")
    print(f"\t> expected: {plaquette_links[k]}")

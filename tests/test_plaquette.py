import sys
if '..' not in sys.path:
    sys.path.append('..')

import logging as log

from group import DihGroup, DihIrreps
from hamiltonian.plaquette import PlaquetteMels, get_plaq_links
from tests.lattice_2x2 import vertices, plaqs_vertices

log.basicConfig(level=log.INFO)

group = DihGroup(4)
irreps = DihIrreps(group.N)

# # Compute plaquette mels
# plaq_matrix = wl_matrix_multiproc(group, irreps, 4, pool_size=8)
# plaq = Plaquette(from_dict=plaq_matrix)

plaq = PlaquetteMels(irreps=irreps, from_file="../pickled/plaquette_data_D4.pkl")
print('> Plaquette loaded')
print(f'\t> #rows: {len(plaq)}')
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

# expected plaquettes links
expect_plq_links = [
    (0, 1, 2, 3),
    (4, 3, 5, 1),
    (5, 7, 4, 6),
    (2, 6, 0, 7)
]

# Testing function plaq_links
print("\n\nPlaquettes links:")
for k, p_vertices in enumerate(plaqs_vertices):
    p_links = get_plaq_links(vertices, p_vertices)
    print(f"> Plaq n. {k}:")
    print(f"\t> computed: {p_links}")
    print(f"\t> expected: {expect_plq_links[k]}")

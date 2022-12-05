import sys
if '..' not in sys.path:
    sys.path.append('..')

from itertools import chain
from group import DihGroup, DihIrreps
from basis import Basis, State
from lattice_2x2 import vertices, nlinks, plaqs_vertices
from hamiltonian.plaquette import get_plaq_links
from basis.contractions import tensor_around_plaq

def compare(statement, message):
    print(f'> {message}:  ', end='')
    if statement:
        print('pass ✓')
    else:
        print('fail ✗')

plaqs_links = [get_plaq_links(vertices, plaq) for plaq in plaqs_vertices]

def flip_plaq(n, p):
    plaq = plaqs_links[p]
    return tuple(n if link in plaq else 0 for link in range(nlinks))

# Setup
print('>> Setting up (group, irreps and physical basis)')
group = DihGroup(4)
irreps = DihIrreps(group.N)
basis = Basis(group, irreps, vertices, nlinks)
print()

# Candidate states
ket_easy    = State((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0))
ket_medium1 = State(flip_plaq(4, 0), (0, 0, 0, 0))
ket_medium2 = State(flip_plaq(4, 2), (0, 0, 0, 0))
ket_hard    = State((4, 4, 4, 4, 4, 4, 4, 4), (0, 0, 0, 0))
shapes_easy    = [(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)]
shapes_medium1 = [(2, 2, 1, 1), (1, 2, 2, 1), (1, 1, 2, 2), (2, 1, 1, 2)]
shapes_medium2 = [(1, 1, 2, 2), (2, 1, 1, 2), (2, 2, 1, 1), (1, 2, 2, 1)]
shapes_hard    = [(2, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2)]

kets = (ket_easy, ket_medium1, ket_medium2, ket_hard)
kets_shapes = (shapes_easy, shapes_medium1, shapes_medium2, shapes_hard)
for ket, shapes in zip(kets, kets_shapes):
    # Check if correct size
    print(f'>> State: {ket}')
    for vertex, shape in enumerate(shapes):
        compare(basis(ket)[vertex].shape == shape, f"Comparing shapes for vertex n. {vertex}")
    for n, plaquette in enumerate(plaqs_vertices):
        tensor = tensor_around_plaq(basis, ket, plaquette)
        expect_tensor_shape = tuple(chain.from_iterable(shapes[v] for v in plaquette))
        compare(tensor.shape == expect_tensor_shape, f"Comparing shapes for plaquette n. {n}")
    print()



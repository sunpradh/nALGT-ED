from group import Group_elem
from collections.abc import Sequence, Callable

IrrepFn = Callable[[Group_elem], float | Sequence[float]]
IrrepIndex = int
IrrepConf = tuple[IrrepIndex, ...]
Vector = Sequence[float]
InvariantSpace = list[Vector]
# Group tuple (g_1, g_2, g_3, g_4)
GroupTuple = tuple[Group_elem, Group_elem, Group_elem, Group_elem]

# Each link is labeled by an int
Link = int
# Each vertex is specified a 4-tuple of links,
# from the right in the counterclockwise direction
VertexLinks = tuple[Link, Link, Link, Link]
# Each vertex of a lattice is labeled by an int
Vertex = int
# Each plaquette is specified by a list of 4 vertices,
# from the bottom left in the counterclockwise direction
PlaqVertices = tuple[Vertex, Vertex, Vertex, Vertex]

# Index format in the irrep basis (j, m, n)
MelIndex = tuple[int, int, int]
# Index format for a plaquette state (j_1, m_1, n_1; ...; j_4, m_4, n_4)
PlaqIndex = tuple[MelIndex, MelIndex, MelIndex, MelIndex]

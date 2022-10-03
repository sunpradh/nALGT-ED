from group import Group_elem
from typing import Iterable, Callable

IrrepFn = Callable[[Group_elem], float | Iterable[float]]
IrrepIndex = int
Link = int
IrrepConf = tuple[IrrepIndex, ...]
VertexLinks = tuple[Link, Link, Link, Link]
Vector = Iterable[float]
InvariantSpace = list[Vector]

# - Index format in the irrep basis (j, m, n)
MelIndex = tuple[int, int, int]
# - Index format for a plaquette state (j_1, m_1, n_1; ...; j_4, m_4, n_4)
PlaqIndex = tuple[MelIndex, MelIndex, MelIndex, MelIndex]
# - Group tuple (g_1, g_2, g_3, g_4)
GroupTuple = tuple[Group_elem, Group_elem, Group_elem, Group_elem]

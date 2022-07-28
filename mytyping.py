from group import Group_elem
from typing import Iterable, Callable

IrrepFn = Callable[[Group_elem], float | Iterable[float]]
IrrepIndex = int
Link = int
IrrepConf = tuple[IrrepIndex, ...]
VertexLinks = tuple[Link, Link, Link, Link]
Vector = Iterable[float]
InvariantSpace = list[Vector]

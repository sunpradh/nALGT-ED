"""
Compute the physical basis, in "compressed form",
for a given group, irreps and lattice geometry
"""
from collections import namedtuple

from basis.invariant import invariant_states
from group import Group, Irreps
from itertools import product
from mytyping import IrrepConf, VertexLinks, InvariantSpace


def vertex_basis(
        group: Group,
        irreps: Irreps,
        state_dict=False,
        sanitized=True
    ) -> dict[IrrepConf, InvariantSpace]:
    """
    Calculate the whole invariant space of a vertex, given `group` and `irreps`.
    Used in building the complete physical Hilbert space
    """
    basis = dict()
    irrep_conf = product(range(len(irreps)), repeat=4)
    for conf in irrep_conf:
        inv_states = invariant_states(
            group,
            irreps,
            conf,
            sanitized=sanitized,
            state_dict=state_dict
        )
        if inv_states:
            shape = tuple(irreps.dim(j) for j in conf)
            inv_states = [v.reshape(shape) for v in inv_states]
            basis.update({conf: inv_states})
    return basis


def vertex_conf(
        irrep_conf: IrrepConf,
        vertex: VertexLinks
    ) -> IrrepConf:
    return tuple(irrep_conf[l] for l in vertex)


StateLabel = namedtuple('StateIndex', ['irreps', 'subindex'])


class Basis:
    def __init__(
            self,
            group: Group,
            irreps: Irreps,
            vertices: list[VertexLinks],
            nlinks: int
        ):
        self.group = group
        self.irreps = irreps
        self.vertices = vertices
        self.nlinks = nlinks
        self._basis = self._compute_basis()
        self.states = self._expand_state_labels()


    def _compute_basis(self) -> dict[IrrepConf, list[InvariantSpace]]:
        """
        Compute the physical Hilbert space, given `group` and `irreps`, the links
        of each vertex (`vertices`) and the number of links
        """
        possible_irrep_confs = product(range(len(self.irreps)), repeat=self.nlinks)
        vbasis = vertex_basis(self.group, self.irreps)
        basis = dict()
        for conf in possible_irrep_confs:
            inv_spaces = [
                vbasis[vertex_conf(conf, vertex)]
                for vertex in self.vertices
                if vertex_conf(conf, vertex) in vbasis
            ]
            if len(inv_spaces) == len(self.vertices):
                basis.update({conf: inv_spaces})
        return basis

    def _expand_state_labels(self):
        states_list = []
        for irrep_conf in self._basis.keys():
            iterators = [range(len(vs)) for vs in self._basis[irrep_conf]]
            states_list += [
                StateLabel(irrep_conf, subindex)
                for subindex in product(*iterators)
            ]
        return states_list

    def __call__(self, state: StateLabel):
        return [self._basis[state.irreps][k] for k in state.subindex]

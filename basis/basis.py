"""
Compute the physical basis, in "compressed form",
for a given group, irreps and lattice geometry
"""
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
            basis.update({conf: inv_states})
    return basis


def vertex_conf(
        irrep_conf: IrrepConf,
        vertex: VertexLinks
    ) -> IrrepConf:
    return tuple(irrep_conf[l] for l in vertex)


def physical_basis(
        group: Group,
        irreps: Irreps,
        vertices: list[VertexLinks],
        nlinks: int
    ) -> dict[IrrepConf, list[InvariantSpace]]:
    """
    Compute the physical Hilbert space, given `group` and `irreps`, the links
    of each vertex (`vertices`) and the number of links
    """
    possible_irrep_confs = product(range(len(irreps)), repeat=nlinks)
    vbasis = vertex_basis(group, irreps)
    basis = dict()
    for conf in possible_irrep_confs:
        inv_spaces = [
            vbasis[vertex_conf(conf, vertex)]
                for vertex in vertices
                if vertex_conf(conf, vertex) in vbasis
        ]
        if len(inv_spaces) == len(vertices):
            basis.update({conf: inv_spaces})
    return basis

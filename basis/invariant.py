"""
Find the invariant states of a single vertex
"""
import numpy as np

from group import Group, Irreps
from basis.gauss import gauss_operator
from utils.mytyping import IrrepConf, IrrepFn, Vector
from utils.linalg import projector, null_space_system
from utils.utils import  sanitize, multiindex


def irrep_conf(conf: IrrepConf, irreps: Irreps) -> tuple[IrrepFn, ...]:
    return tuple(irreps(c) for c in conf)


def size_conf(conf: IrrepConf, irreps: Irreps) -> tuple[int, ...]:
    return tuple(irreps.dim(c) for c in conf)


def multiind_conf(index: int, conf: IrrepConf, irreps: Irreps) -> tuple[int, ...]:
    return multiindex(
                index,
                size_conf(conf, irreps),
                len(conf),
                conf
            )


# TODO: to_state_dict does not belongs to this module
def to_state_dict(
        state: Vector,
        conf: IrrepConf,
        irreps: Irreps
    ) -> dict[tuple[int, ...], float]:
    non_zero_indices = np.nonzero(state)[0]
    state_dict = {
            multiind_conf(index, conf, irreps): state[index]
                for index in non_zero_indices
                }
    return state_dict


# TODO: transformation to a state_dict should be separate
#       from the calculation of the invariant_space
def invariant_states(
        group: Group,
        irreps: Irreps,
        conf: tuple[int],
        sanitized=True,
        state_dict=True
    ):
    gauss_null_space = null_space_system([
                projector(gauss_operator(g, irrep_conf(conf, irreps)))
                for g in group.generators
            ])
    f = lambda x: sanitize(x) if sanitized else x
    g = lambda x: to_state_dict(x, conf, irreps) if state_dict else x
    return [g(f(state)) for state in gauss_null_space.T]

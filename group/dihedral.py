"""
D4 digital simulation
"""
import math
import numpy as np
from .base_group import Group_elem, Group, Irreps


def Dih_elem(N):
    class Dihedral_element(Group_elem):
        __slots__ = "r", "s"

        def __init__(self, r, s):
            """ All the elements of DN are representated as r^k * s"""
            self.r = r % N
            self.s = s % 2

        def __eq__(self, other):
            return self.r == other.r and self.s == other.s

        def __mul__(self, other):
            if self.s == 1:
                return Dihedral_element(
                        self.r + (N - other.r),
                        (self.s + other.s) % 2
                        )
            else:
                return Dihedral_element(self.r + other.r, other.s)

        def __invert__(self):
            if self.s == 0:
                return Dihedral_element(N - self.r, 0)
            else:
                return self

        def __hash__(self):
            return hash((N, self.r, self.s))

        def __repr__(self):
            return f"D{N}({self.r},{self.s})"

        def __str__(self):
            return f"({self.r},{self.s})"

    return Dihedral_element


class Dih_group(Group):
    __slots__ = "N", "name", "elements"

    def __init__(self, N):
        self.N = N
        self.name = f"D{self.N}"
        self.elements = [
            Dih_elem(N)(r, s) for s in [0, 1] for r in range(N)
        ]

    def __len__(self):
        return 2*self.N

    def elem(self, index):
        return self.elements[index[0] + self.N*index[1]]

    @property
    def id(self):
        """Returns the identity element of the group"""
        return self.elem((0, 0))

    @property
    def r(self):
        """Returns the generator of the rotations of the group"""
        return self.elem((1, 0))

    @property
    def s(self):
        """Returns the generator of the reflection of the group"""
        return self.elem((0, 1))

    @property
    def generators(self):
        """Returns a list of the generators"""
        return [self.r, self.s]


_Dih_even_1d_irreps = [
    lambda g: 1,
    lambda g: (-1)**g.r,
    lambda g: (-1)**g.s,
    lambda g: (-1)**g.r * (-1)**g.s
]

_Dih_odd_1d_irreps = [
    lambda g: 1,
    lambda g: (-1)**g.s
]


def _Dih_2d_irrep_real(N, k):
    a = 2 * math.pi * k / N

    def irrep(g):
        c = math.cos(a*g.r)
        s = math.sin(a*g.r)
        if g.s == 0:
            return np.array([[c, -s], [s, c]])
        else:
            return np.array([[c, s], [s, -c]])

    return irrep

def _Dih_2d_irrep_complex(N, k):
    phase = np.exp(2j * math.pi / N)

    def irrep(g):
        if g.s == 0:
            return np.array([[phase**(k*g.r), 0], [0, phase**(-k*g.r)]])
        else:
            return np.array([[0, phase**(-k*g.r)], [phase**(k*g.r), 0]])

    return irrep


class Dih_Irreps(Irreps):
    __slots__ = "N", "_1d_irreps", "_2d_irreps", "irreps", "chars"

    def __init__(self, N, complex=False):
        self.N = N
        Dih_2d_irrep = _Dih_2d_irrep_complex if complex else _Dih_2d_irrep_real
        if N % 2 == 0:
            self._1d_irreps = _Dih_even_1d_irreps.copy()
            self._2d_irreps = [Dih_2d_irrep(N, k) for k in range(1, N//2)]
        else:
            self._1d_irreps = _Dih_odd_1d_irreps.copy()
            self._2d_irreps = [Dih_2d_irrep(N, k) for k in range(1, (N+1)//2)]
        super().__init__()

    def __repr__(self):
        return f"<D{self.N} irreps: " + \
            f"{len(self._1d_irreps)} <1d reps> + {len(self._2d_irreps)} <2d reps>>"

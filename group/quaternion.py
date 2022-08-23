import re
import numpy as np
from itertools import product
from .base_group import Group_elem, Group, Irreps


# Q8 simplified multiplication table, without the signs
Q8_elem_table = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
]

# Q8 simplified multiplication table, only the signs
Q8_sign_table = [
    [+1, +1, +1, +1],
    [+1, -1, +1, -1],
    [+1, -1, -1, +1],
    [+1, +1, -1, -1]
]

def sign_str(sign):
    return {-1: '-', +1: '+'}[sign]

def elem_int(elem_str):
    return {'1': 0, 'i': 1, 'j': 2, 'k': 3}[elem_str]

def elem_str(elem_int):
    return ['1', 'i', 'j', 'k'][elem_int]


class Q8_elem(Group_elem):
    __slots__ = 'sign', 'elem'

    def __init__(self, input: str):
        if match := re.fullmatch("([+-]?)([1ijk])", input):
            self.sign = -1 if match.group(1) == "-" else +1
            self.elem = elem_int(match.group(2))
        else:
            raise ValueError("Invalid input string \"{input}\"")

    def __mul__(self, other):
        # Group multiplication implemented with a lookup table because
        # I could not think of a better method
        result_sign = self.sign * other._sign * Q8_sign_table[self.elem][other._elem]
        result_elem = Q8_elem_table[self.elem][other._elem]
        return Q8_elem(f"{sign_str(result_sign)}{elem_str(result_elem)}")

    def __invert__(self):
        return Q8_elem(f"{sign_str(-1*self.sign)}{self._elem_str()}")

    def __hash__(self):
        return hash(('Q8', self.sign, self.elem))

    def __repr__(self):
        return f"Q8({sign_str(self.sign)}{elem_str(self.elem)})"

    def __str__(self):
        return f"{sign_str(self.sign)}{elem_str(self.elem)}"


class Q8(Group):
    __slots__ = "name", "elements"

    def __init__(self):
        self.name = "Q8"
        self.elements = [
            Q8_elem(f"{sign}{elem}") for sign, elem in product("+-", "1ijk")
        ]

    def __len__(self):
        return len(self.elements)


    @property
    def id(self):
        return self.elements[0]

    def elem(self, index):
        return self.elements[index]

    @property
    def generators(self):
        return self.elements[:4]


pauli = {
    '1': np.array([[1, 0],   [0,  1]]),
    'x': np.array([[0, 1],   [1,  0]]),
    'y': np.array([[0, -1j], [1j, 0]]),
    'z': np.array([[1, 0],   [0,  -1]]),
}

def _make_1d_irreps(elem: int):
    return lambda g: -1 if g.elem == elem else 1

def _Q8_2d_irrep(g: Q8_elem):
    match g.elem:
        case 0: return g.sign * pauli['1']
        case 1: return -1j * g.sign * pauli['x'] # i
        case 2: return -1j * g.sign * pauli['y'] # j
        case 3: return -1j * g.sign * pauli['z'] # k

class Q8_Irreps(Irreps):
    __slots__ = "_1d_irreps", "_2d_irreps", "irreps", "chars"

    def __init__(self):
        self._1d_irreps = [
            lambda g: 1,
            _make_1d_irreps(1), # -1 only on "+-i"
            _make_1d_irreps(2), # -1 only on "+-j"
            _make_1d_irreps(3)  # -1 only on "+-k"
        ]
        self._2d_irreps = [ _Q8_2d_irrep ]
        super().__init__()

    def __repr__(self):
        return "<Q8 irreps: " + \
            f"{len(self._1d_irreps)} <1d reps> + {len(self._2d_irreps)} <2d reps>>"

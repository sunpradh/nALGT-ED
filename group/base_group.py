"""Metaclasses for group element, group and irriducible representation (irreps)"""

from abc import ABC, abstractmethod
from itertools import product
import numpy as np

# TODO: DOCUMENTATION

class Group_elem(ABC):
    @abstractmethod
    def __mul__(self):
        pass

    @abstractmethod
    def __invert__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Group(ABC):
    __slots__ = "name", "elements"

    def __init__(self):
        self.name = "(undefined)"
        self.elements = []

    def __repr__(self):
        return f"<{self.name} group>"

    def __str__(self):
        return self.name

    @abstractmethod
    def __len__(self):
        """Order of the group"""
        pass

    @abstractmethod
    def elem(self, index):
        pass

    @property
    @abstractmethod
    def id(self):
        """Identity element of the group"""
        pass

    @property
    @abstractmethod
    def generators(self):
        """List of generators of the group"""
        pass

    def mul_table(self):
        """Returns the multiplication table of the group"""
        return [[g*h for h in self.elements] for g in self.elements]

    def conj_class(self, g):
        """Return the conjugacy class of the element `g`"""
        cclass = list({(~h) * g * h for h in self.elements})
        return cclass

    def conj_classes(self):
        """Return a list of all the conjugacy classes of the group"""
        cclasses = []
        for g in self.elements:
            cclass = self.conj_class(g)
            if cclass not in cclasses:
                cclasses.append(cclass)
        return cclasses

    def __iter__(self):
        return iter(self.elements)



class Irreps(ABC):
    __slots__ = "_1d_irreps", "_2d_irreps", "irreps", "chars"

    def __init__(self):
        if not hasattr(self, "_1d_irreps") or not hasattr(self, "_2d_irreps"):
            raise RuntimeError("Irreps are not defined")
        elif not (isinstance(self._1d_irreps, list) and isinstance(self._2d_irreps, list)):
            raise RuntimeError("Irreps not passed as lists")
        self.irreps = self._1d_irreps + self._2d_irreps
        self._make_chars()

    def _make_chars(self):
        self.chars = self._1d_irreps.copy()
        # list comprehension gives problems because it chaches the iterator
        # solvables with map, see
        # stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
        self.chars += list(map(
                            lambda f: lambda g: f(g).trace(),
                            self._2d_irreps
                        ))

    @abstractmethod
    def __repr__(self):
        pass

    def __call__(self, k):
        return self.irreps[k]

    def __iter__(self):
        return iter(self.irreps)

    def __len__(self):
        return len(self.irreps)

    @property
    def abelian(self):
        """One-dimensional irreps"""
        return self.irreps[:len(self._1d_irreps)]

    @property
    def non_abelian(self):
        """Two-dimensional irreps"""
        return self.irreps[len(self._1d_irreps):]

    def dim(self, j=None):
        """
        If no parameter `j` is given, it return the total numbers of matrix elements
        Otherwise, it return the dimension of the `j`-th irrep
        """
        if j > len(self.irreps) or j < 0:
            raise ValueError(f"There is no j={j} representation")
        elif j is None:
            return len(self._1d_irreps) + 4*len(self._2d_irreps)
        elif j < len(self._1d_irreps):
            return 1
        else:
            return 2

    def mel(self, j, m, n):
        """
        Return the matrix elements (`m`,`n`) of the `j`-th representation
        as a function of group elements
        """
        if self.dim(j) == 1:
            return lambda g: self.irreps[j](g)
        if self.dim(j) == 2:
            return lambda g: self.irreps[j](g)[m, n]

    def mels(self, g):
        """Return all the matrix elements of the element `g`"""
        elems_1d = np.array([irr(g) for irr in self._1d_irreps])
        elems_2d = np.concatenate([irr(g).ravel() for irr in self._2d_irreps])
        return np.concatenate((elems_1d, elems_2d))


    def mels_dict(self, g):
        """Return the labels of the matrix elements"""
        mat_elems = dict()
        for j, irr in enumerate(self.irreps):
            r = irr(g)
            if np.shape(r):
                for m, n in product(*map(range, r.shape)):
                    mat_elems.update({f"{j}{m}{n}": r[m,n]})
            else:
                mat_elems.update({f"{j}": r })
        return mat_elems

    def mel_indices(self):
        return [ (j, m, n)
            for j in range(len(self.irreps))
            for m, n in product(range(self.dim(j)), repeat=2)
        ]


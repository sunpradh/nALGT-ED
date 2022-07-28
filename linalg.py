"""Linear algebra useful functions"""

import numpy as np
from functools import reduce
from scipy.linalg import null_space

def common_dtype(A, B, default=np.float64):
    """Return a common numpy dtype for given objects A and B"""
    if hasattr(A, 'dtype') and hasattr(B, 'dtype'):
        dtype = A.dtype or B.dtype
    else:
        dtype = default
    return dtype

def shape(A):
    """Returns the shape of A or (1,1) if is just a number (instead of an empty tuple)"""
    return np.shape(A) if np.shape(A) else (1,1)


def projector(op, val=1.0):
    """Given an object `op` returns the projector `op - val*identity`"""
    return op - val * np.eye(shape(op)[0])


def matrix_system(A, B):
    """Given two matrices `A` and `B` returns a matrix with the rows of A over the rows of B"""
    if shape(A)[1] != shape(B)[1]:
        return ValueError("Matrices do not have matching dimensions," + \
                f"{A.shape} and {B.shape}")
    dtype = common_dtype(A, B)
    system = np.ndarray((shape(A)[0]+shape(B)[0], shape(A)[1]), dtype=dtype)
    system[:shape(A)[0], :] = A
    system[shape(A)[0]:, :] = B
    return system


def null_space_system(matrices):
    """Returns the vectors of the common null space of a list of matrices"""
    return null_space(reduce(matrix_system, matrices))


def direct_sum(A, B):
    """Returns the direct sum of A and B"""
    dtype = common_dtype(A, B)
    result = np.zeros(np.add(shape(A), shape(B)), dtype=dtype)
    result[:shape(A)[0], :shape(A)[0]] = A
    result[shape(A)[1]:, shape(A)[1]:] = B
    return result


def left_tensor(A):
    """Given a matrix `A` returns the matrix `A \otimes Identity`"""
    return np.kron(A, np.eye(shape(A)[0]))


def right_tensor(A):
    """Given a matrix `A` returns the matrix `Identity \otimes A`"""
    return np.kron(np.eye(shape(A)[0]), A)


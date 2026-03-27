import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    rows, cols = A.shape
    result = np.empty((cols, rows), dtype=A.dtype)
    for i in range(rows):
        result[:, i] = A[i, :]
    return result
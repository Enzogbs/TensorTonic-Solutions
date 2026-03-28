import numpy as np

def _L2_norm(x, axis=1):
    """
    L2 Normalization
    """
    return np.sqrt(np.sum(x ** 2, axis=axis))

def _L1_norm(x, axis=1):
    """
    L1 Normalization
    """
    return np.sum(np.abs(x), axis=axis)

def _max_norm(x, axis=1):
    """
    Max Normalization
    """
    return np.max(x, axis=axis)

def matrix_normalization(matrix, axis=0, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.asarray(matrix, dtype=float)
    
    if matrix.ndim != 2:
        return None
    
    norms_fn = {
                "l2": _L2_norm,
                "l1": _L1_norm,
                "max": _max_norm
                }
    
    norm = norms_fn.get(norm_type)

    if norm is None:
        return None

    if axis is None:
        flat = matrix.flatten().reshape(1, -1)
        norm = norm(flat, axis=1)[0]
        if norm == 0:
            return matrix
        return matrix / norm
    
    if not isinstance(axis, int) or axis >= matrix.ndim:
        return None
    
    norm = norm(matrix, axis=axis)
    norm = np.where(norm == 0, 1, norm)

    return matrix / np.expand_dims(norm, axis=axis)


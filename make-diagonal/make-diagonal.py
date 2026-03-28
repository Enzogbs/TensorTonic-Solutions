import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    v = np.array(v)
    matrix = np.eye(v.shape[0])
    
    return matrix * v

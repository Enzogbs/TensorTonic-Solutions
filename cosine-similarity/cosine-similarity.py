import numpy as np
from numpy import linalg as LA

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = LA.norm(a), LA.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return np.dot(a, b) / ( norm_a * norm_b )
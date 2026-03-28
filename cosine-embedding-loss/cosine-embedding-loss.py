import numpy as np
import numpy.linalg as LA

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1, x2 = np.array(x1), np.array(x2)
    
    norm1, norm2 = LA.norm(x1), LA.norm(x2)
    if norm1 == 0 or norm2 == 0:
        return None
    
    cos = np.dot(x1 / norm1, x2 / norm2)
    
    if label == 1:
        return 1 - cos
    elif label == -1:
        return max(0, cos - margin)
    else:
        return None
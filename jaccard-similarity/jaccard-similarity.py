import numpy as np

def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    if set_a is None or set_b is None:
        return None
    
    set_a, set_b = np.array(set_a), np.array(set_b)
    union = np.union1d(set_a, set_b).size

    if union == 0.0:
        return 0.0
    
    return np.intersect1d(set_a, set_b).size / union
import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    if k > len(recommended):
        return None
    recommended, relevant = np.array(recommended[:k]), np.array(relevant)

    
    
    precision = np.intersect1d(recommended, relevant).size / k
    recall = np.intersect1d(recommended, relevant).size / relevant.size

    return [precision, recall]
    
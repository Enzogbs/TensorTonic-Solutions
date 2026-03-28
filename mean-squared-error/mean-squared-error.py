import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)

    diff = y_pred - y_true
    return np.mean(diff ** 2)

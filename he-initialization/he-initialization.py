import numpy as np
import math

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    W = np.array(W)
    L = math.sqrt(6 / fan_in)
    W = W * 2 * L - L
    return W
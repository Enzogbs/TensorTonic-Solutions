import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.array(g)

    if max_norm <= 0:
        return g
        
    g_norm = np.sqrt(np.sum(g ** 2))
    
    return g if g_norm <= max_norm else g * max_norm / g_norm
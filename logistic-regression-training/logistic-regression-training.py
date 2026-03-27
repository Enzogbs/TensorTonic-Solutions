import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _binary_cross_entropy(y, p):
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X, y = np.array(X), np.array(y)
    
    w = np.zeros(X.shape[1])
    b = 0.0
    
    for step in range(0, steps):
        z = X @ w + b
        p = _sigmoid(z)
        loss = _binary_cross_entropy(y, p)
        dw = (1/y.size) * X.T @ (p - y) 
        db = np.mean(p - y) 
        
        w -= lr * dw
        b -= lr * db
    return w, b
        
        
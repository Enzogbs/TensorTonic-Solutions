def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    
    for step in range(0, steps):
        dx = 2 * a * x + b
        x -= lr * dx
    return x
        
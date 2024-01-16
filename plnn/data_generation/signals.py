import numpy as np

def get_binary_function(tcrit, p0, p1):
    """Return a binary function that changes from p0 to p1 at time tcrit."""
    return lambda t: p0 * (t < tcrit) + p1 * (t >= tcrit)

def get_sigmoid_function(tcrit, p0, p1, r):
    """Return a sigmoid function that changes from p0 to p1 at time tcrit."""
    return lambda t: p0 + 0.5*(p1 - p0) * (1 + np.tanh(r*(t-tcrit)))

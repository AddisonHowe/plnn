import numpy as np
import jax.numpy as jnp

def get_binary_function(tcrit, s0, s1):
    """Return a binary function that changes from s0 to s1 at time tcrit."""
    return jnp.vectorize(
        lambda t: s0 * (t < tcrit) + s1 * (t >= tcrit),
        signature='()->(n)',
    )

def get_sigmoid_function(tcrit, s0, s1, r):
    """A sigmoid signal function.
    Args:
        tcrit (Array[float]) : Critical times of each signal. At time tcrit[i] 
            the ith signal is the average of s0[i] and s1[i]. Shape (nsignals,)
        s0 (Array[float]) : Inititial values of each signal. Shape (nsignals,)
        s1 (Array[float]) : Final values of each signal. Shape (nsignals,)
        r (Array[float]) : Change rate of each signal. Shape (nsignals,)
    Returns:
        (callable) : A function taking a scalar input, time, and returning 
            an array of shape (nsignals,) containing the signal value at that 
            time. If the input is an ndarray of shape (*), the signal is 
            computed in a vectorized fashion, with output shape (*, nsignals).
    """
    return jnp.vectorize(
        lambda t: s0 + 0.5*(s1 - s0) * (1 + jnp.tanh(r*(t-tcrit))), 
        signature='()->(n)',
    )

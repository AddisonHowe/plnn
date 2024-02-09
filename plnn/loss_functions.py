"""Loss functions for model training.

Provides functions:
    select_loss_function
    kl_divergence_loss
    mean_cov_loss
    mean_diff_loss
"""

import jax
import jax.numpy as jnp

def select_loss_function(key)->callable:
    """Loss function selector method.
    
    Args:
        key (str) : loss function identifier.
    
    Returns:
        callable : Loss function, taking as inputs simulated and observed data.
    """
    if key == 'kl':
        return kl_divergence_loss
    elif key == 'mcd':
        return mean_cov_loss
    elif key == 'md':
        return mean_diff_loss
    else:
        msg = f"Unknown loss function identifier {key}."
        raise RuntimeError(msg)


######################
##  Loss Functions  ##
######################
    
def kl_divergence_loss(q_samps, p_samps) -> float:
    """Estimate the KL divergence. Returns the average over all batches.

    Adapted from:
      https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    Args:
        q_samp (array) : Batched samples from estimated (i.e. approximate) 
            distribution. Shape (b,m,d).
        p_samp (array) : Batched samples from target (i.e. true) distribution. 
            Shape (b,n,d).
    Returns:
        (float) KL estimate of D(P||Q), averaged across batches.
    """
    return jnp.mean(jax.vmap(kl_divergence_est)(q_samps, p_samps))

def mean_cov_loss(y_sim, y_obs) -> float:
    """Loss function based on the difference of first and second moments.

    Args:
        y_sim (Array): Batched samples from simulation. Shape (b,m,d).
        y_obs (Array): Batched samples from observations. Shape (b,n,d).

    Returns:
        (float): Loss value, averaged across batches.
    """
    mu_sim = jnp.mean(y_sim, axis=1)
    mu_obs = jnp.mean(y_obs, axis=1)
    cov_sim = batch_cov(y_sim)
    cov_obs = batch_cov(y_obs)
    mu_err = jnp.sum(jnp.square(mu_sim - mu_obs), axis=1)
    cov_err = jnp.sum(jnp.square(cov_sim - cov_obs), axis=(1,2))
    return jnp.mean(mu_err + cov_err)

def mean_diff_loss(y_sim, y_obs) -> float:
    """Loss function based on the difference of first moments.

    Args:
        y_sim (Array): Batched samples from simulation. Shape (b,m,d).
        y_obs (Array): Batched samples from observations. Shape (b,n,d).

    Returns:
        (float): Loss value, averaged across batches.
    """
    mu_sim = jnp.mean(y_sim, axis=1)
    mu_obs = jnp.mean(y_obs, axis=1)
    mu_err = jnp.sum(jnp.square(mu_sim - mu_obs), axis=1)
    return jnp.mean(mu_err)


########################
##  Helper Functions  ##
########################

def euclidean_distance(x, y):
    return jnp.sqrt(jnp.sum(jnp.square(x - y)))

def batch_cov(batch_points):
    """
    Returns:
        Shape (b,d,d) tensor
    """
    return jax.vmap(jnp.cov, 0)(batch_points.transpose((0, 2, 1)))

def cdist(x, y):
    return jax.vmap(lambda x1: jax.vmap(
        lambda y1: euclidean_distance(x1, y1))(y))(x)

def kl_divergence_est(q_samp, p_samp):
    """Estimate the KL divergence.
    
    Adapted from:
      https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    Args:
        q_samp (array) : Samples from estimated (i.e. approximate) distribution.
            Shape (m,d).
        p_samp (array) : Samples from target (i.e. true) distribution. 
            Shape (n,d).
    Returns:
        (float) KL estimate of D(P||Q).
    """

    n, d = p_samp.shape
    m, _ = q_samp.shape
    
    diffs_xx = cdist(p_samp, p_samp)
    diffs_xy = cdist(p_samp, q_samp)
    
    r = -jax.lax.top_k(-diffs_xx, 2)[0][:,1]
    s = -jax.lax.top_k(-diffs_xy, 1)[0][:,0]
    lossval = -jnp.log(r/s).sum() * d/n + jnp.log(m / (n - 1.))
    return lossval

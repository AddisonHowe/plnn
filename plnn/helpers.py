"""General Helper Functions
"""

import jax
import jax.numpy as jnp

@jax.jit
def mean_cov_loss(y_sim, y_obs):
    mu_sim = jnp.mean(y_sim, axis=1)
    mu_obs = jnp.mean(y_obs, axis=1)
    cov_sim = batch_cov(y_sim)
    cov_obs = batch_cov(y_obs)
    mu_err = jnp.sum(jnp.square(mu_sim - mu_obs), axis=1)
    cov_err = jnp.sum(jnp.square(cov_sim - cov_obs), axis=(1,2))
    return jnp.mean(mu_err + cov_err)

@jax.jit
def mean_diff_loss(y_sim, y_obs):
    mu_sim = jnp.mean(y_sim, axis=1)
    mu_obs = jnp.mean(y_obs, axis=1)
    mu_err = jnp.sum(jnp.square(mu_sim - mu_obs), axis=1)
    return jnp.mean(mu_err)

@jax.jit
def batch_cov(batch_points):
    """
    Returns:
        Shape (b,d,d) tensor
    """
    return jax.vmap(jnp.cov, 0)(batch_points.transpose((0, 2, 1)))

@jax.jit
def euclidean_distance(x, y):
    return jnp.sqrt(jnp.sum(jnp.square(x - y)))

@jax.jit
def cdist(x, y):
    return jax.vmap(lambda x1: jax.vmap(
        lambda y1: euclidean_distance(x1, y1))(y))(x)

@jax.jit
def kl_divergence_est(q_samp, p_samp):
    """Estimate the KL divergence. Returns the average over all batches.
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

@jax.jit
def kl_divergence_loss(q_samps, p_samps):
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

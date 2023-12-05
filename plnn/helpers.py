"""General Helper Functions
"""

import jax
import jax.numpy as jnp

def mean_cov_loss(y_sim, y_obs):
    mu_sim = jnp.mean(y_sim, axis=1)
    mu_obs = jnp.mean(y_obs, axis=1)
    cov_sim = batch_cov(y_sim)
    cov_obs = batch_cov(y_obs)
    mu_err = jnp.sum(jnp.square(mu_sim - mu_obs), axis=1)
    cov_err = jnp.sum(jnp.square(cov_sim - cov_obs), axis=(1,2))
    return jnp.mean(mu_err + cov_err)

def mean_diff_loss(y_sim, y_obs):
    mu_sim = jnp.mean(y_sim, axis=1)
    mu_obs = jnp.mean(y_obs, axis=1)
    mu_err = jnp.sum(jnp.square(mu_sim - mu_obs), axis=1)
    return jnp.mean(mu_err)

def batch_cov(batch_points):
    """
    Returns:
        Shape (b,d,d) tensor
    """
    return jax.vmap(jnp.cov, 0)(batch_points.transpose((0, 2, 1)))

def kl_divergence_est(q_samp, p_samp):
    """Estimate the KL divergence. Returns the average over all batches.
    Adapted from:
      https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    Args:
        q_samp : Estimated sample distribution of shape (b,m,d)
        p_samp : Target sample distribution of shape (b,n,d)
    Returns:
        (float) KL estimate D(p|q), averaged over each batch.
    """

    # _, n, d = p_samp.shape
    # _, m, _ = q_samp.shape
    
    # diffs_xx = torch.cdist(p_samp, p_samp, p=2, 
    #                        compute_mode='donot_use_mm_for_euclid_dist')  
    # diffs_xy = torch.cdist(q_samp, p_samp, p=2, 
    #                        compute_mode='donot_use_mm_for_euclid_dist')
    
    # r = torch.kthvalue(diffs_xx, 2, dim=1)[0]
    # s = torch.kthvalue(diffs_xy, 1, dim=1)[0]

    # vals = -torch.log(r/s).sum(axis=1) * d/n + np.log(m/(n-1.))
    # return torch.mean(vals)
    raise NotImplementedError()

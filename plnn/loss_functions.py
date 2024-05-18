"""Loss functions for model training.

Provides functions:
    select_loss_function
    kl_divergence_loss
    mean_cov_loss
    mean_diff_loss
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

def select_loss_function(func_id, **kwargs)->callable:
    """Loss function selector method.
    
    Args:
        func_id (str) : loss function identifier.
        
    Keyword Arguments:
        tol (float, kwarg, default 1e-6) : tolerance if using loss function
            kl_divergence_loss_v2.
    
    Returns:
        callable : Loss function, taking as inputs simulated and observed data.
    """
    if func_id == 'kl':
        return kl_divergence_loss
    if func_id == 'klv2':
        tol = kwargs.get('tol', 1e-6)
        def loss_fn(xsim, xobs):
            return kl_divergence_loss_v2(xsim, xobs, tol=tol) 
        return loss_fn
    elif func_id == 'mcd':
        return mean_cov_loss
    elif func_id == 'md':
        return mean_diff_loss
    elif func_id == 'mmd':
        kernel = kwargs.get('kernel', 'multiscale')
        bw_range = kwargs.get('bw_range', None)
        if bw_range is None:
            if kernel == 'multiscale':
                bw_range = jnp.array([0.2, 0.5, 0.9, 1.3])
            elif kernel == 'rbf':
                bw_range = jnp.array([[10, 15, 20, 50]])
        def loss_fn(xsim, xobs):
            return mmd_loss(xsim, xobs, kernel=kernel, bw_range=bw_range) 
        return loss_fn
    else:
        msg = f"Unknown loss function identifier {func_id}."
        raise RuntimeError(msg)


######################
##  Loss Functions  ##
######################
    
def kl_divergence_loss(
        q_samps: Float[Array, "b m d"], 
        p_samps: Float[Array, "b n d"],
) -> Float:
    """Estimate the KL divergence. Returns the average over all batches.

    We make the first argument q, rather than p, since we want our loss 
    functions to take as first argument the approximation distribution.

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
    return jnp.mean(jax.vmap(kl_divergence_est)(p_samps, q_samps))

def kl_divergence_loss_v2(
        q_samps: Float[Array, "b m d"], 
        p_samps: Float[Array, "b n d"],
        tol=1e-6,
) -> Float:
    """Estimate the KL divergence. Returns the average over all batches.

    We make the first argument q, rather than p, since we want our loss 
    functions to take as first argument the approximation distribution.

    Uses a smooth approximation of the minimum to determine the nearest 
    neighbor between points in P and points in Q.

    Args:
        q_samp (array) : Batched samples from estimated (i.e. approximate) 
            distribution. Shape (b,m,d).
        p_samp (array) : Batched samples from target (i.e. true) distribution. 
            Shape (b,n,d).
        tol (float) : Optional, default 1e-6.
    Returns:
        (float) KL estimate of D(P||Q), averaged across batches.
    """
    return jnp.mean(jax.vmap(smooth_kl_est, (0,0,None))(p_samps, q_samps, tol))

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

def mmd_loss(y_sim, y_obs, kernel='multiscale', bw_range=None) -> float:
    """Loss function based on Maximum Mean Discrepancy.

    Args:
        y_sim (Array): Batched samples from simulation. Shape (b,m,d).
        y_obs (Array): Batched samples from observations. Shape (b,n,d).

    Returns:
        (float): Loss value, averaged across batches.
    """
    return jnp.mean(
        jax.vmap(compute_mmd, (0, 0, None, None))(
            y_sim, y_obs, kernel, bw_range
        )
    )

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

def cdist(
        x: Float[Array, "n d"], 
        y: Float[Array, "m d"], 
) -> Float[Array, "n m"]:
    """Return pairwise distances between n and m d-dimensional vectors.

    Args:
        x (Array) : Shape (n,d)
        y (Array) : Shape (m,d)

    Returns:
        (Array) : Shape (n,m). Matrix of pairwise distances with each row 
            corresponding to an element of x and each column to an element of y.
    """
    return jax.vmap(lambda x1: jax.vmap(
        lambda y1: euclidean_distance(x1, y1))(y))(x)

def kl_divergence_est(
        p_samp: Float[Array, "n d"],
        q_samp: Float[Array, "m d"], 
) -> Float:
    """Estimate the KL divergence.
    
    Adapted from:
      https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    Args:
        p_samp (array) : Samples from target (i.e. true) distribution P, and 
            denoted by X. Shape (n,d).
        q_samp (array) : Samples from estimated (i.e. approximate) 
            distribution Q, and denoted by Y. Shape (m,d).
    Returns:
        (float) KL estimate of D(P||Q).
    """

    n, d = p_samp.shape
    m, _ = q_samp.shape
    
    # Compute r(xi) and s(xi) for each of the n points xi in X ~ p(x).
    # r(xi) is the distance of xi to its nearest neighbor in the set X\{xi}.
    # s(xi) is the distance of xi to its nearest neighbor in the set Y.

    # Start by computing pairwise distances d(X,X) and d(X,Y)
    diffs_xx = cdist(p_samp, p_samp)  # diffs_xx[i,j] = d(xi, xj)
    diffs_xy = cdist(p_samp, q_samp)  # diffs_xy[i,j] = d(xi, yj)

    # top_k(D, k)[0] returns an array of shape (D.shape[0], k). Each row 
    # corresponds to a row of D, with top k elements in decreasing order.
    # The double negatives thus smallest k values, in increasing order.
    # For dist(X,X), the smallest will be 0, and we want the second smallest.
    r = -jax.lax.top_k(-diffs_xx, 2)[0][:,1]
    s = -jax.lax.top_k(-diffs_xy, 1)[0][:,0]
    lossval = jnp.log(m/(n-1.)) - jnp.log(r/s).sum() * d/n
    return lossval

def smooth_kl_est(p_samp, q_samp, tol=1e-6):
    """Estimate the KL divergence.
    
    Uses the logsumexp to approximate the minimum distance between points in 
    p_samp and points in q_samp. Uses a tolerance hyperparameter to ensure the
    estimated minimum is within `tol` of the true minimum. This is important, 
    since we cannot have a negative estimate due to the log in the loss.

    Args:
        p_samp (array) : Samples from target (i.e. true) distribution P, and 
            denoted by X. Shape (n,d).
        q_samp (array) : Samples from estimated (i.e. approximate) 
            distribution Q, and denoted by Y. Shape (m,d).
        tol (float) : Tolerance on the estimation error for the nearest 
            neighbor distance. Optional, default 1e-6.
    Returns:
        (float) KL estimate of D(P||Q).
    """
    n, d = p_samp.shape
    m, _ = q_samp.shape

    t = -jnp.log(m) / tol

    diffs_xx = cdist(p_samp, p_samp)  # diffs_xx[i,j] = d(xi, xj)
    r_topk = jax.lax.top_k(-diffs_xx, 2)
    r = -r_topk[0][:,1]

    diffs_xy = cdist(p_samp, q_samp)  # diffs_xy[i,j] = d(xi, yj)
    s = jax.scipy.special.logsumexp(t * diffs_xy, axis=1) / t
    
    lossval = jnp.log(m/(n-1.)) - d/n * jnp.log(r/s).sum()
    return lossval


def _multiscale_kernel(dxx, dyy, dxy, bw):
    a = bw * bw
    xx = a / (a + dxx)
    yy = a / (a + dyy)
    xy = a / (a + dxy)
    return xx, yy, xy

def _rbf_kernel(dxx, dyy, dxy, bw):
    xx = jnp.exp(-0.5 * dxx / bw)
    yy = jnp.exp(-0.5 * dyy / bw)
    xy = jnp.exp(-0.5 * dxy / bw)
    return xx, yy, xy

def compute_mmd(x, y, kernel='multiscale', bw_range=[0.2, 0.5, 0.9, 1.3]):
    """
    """

    n, d = x.shape
    m, _ = y.shape

    cxx = 1. / (n * (n - 1))
    cyy = 1. / (m * (m - 1))
    cxy = -1. / (n * m)

    xx, yy, zz = jnp.dot(x, x.T), jnp.dot(y, y.T), jnp.dot(x, y.T)
    rx = jnp.expand_dims(jnp.diag(xx), 0).repeat(xx.shape[0], axis=0)
    ry = jnp.expand_dims(jnp.diag(yy), 0).repeat(yy.shape[0], axis=0)

    dxx = rx.T + rx - 2. * xx  # squared distances between x and x
    dyy = ry.T + ry - 2. * yy  # squared distances between y and y
    dxy = rx.T + ry - 2. * zz  # squared distances between x and y

    bw_range = jnp.array(bw_range)
    if kernel == "multiscale":
        xxs, yys, xys = jax.vmap(_multiscale_kernel, (None, None, None, 0))(
            dxx, dyy, dxy, bw_range
        )
        
    if kernel == "rbf":
        xxs, yys, xys = jax.vmap(_rbf_kernel, (None, None, None, 0))(
            dxx, dyy, dxy, bw_range
        )

    xxs = jnp.sum(xxs, axis=0)
    yys = jnp.sum(yys, axis=0)
    xys = jnp.sum(xys, axis=0)
    return cxy * xys.sum() \
        + cxx * (xxs.sum() - jnp.trace(xxs)) \
        + cyy * (yys.sum() - jnp.trace(yys))

"""Functions for vector field analysis.

"""

import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import jax.random as jrandom


def estimate_minima(
        model, 
        tilt, 
        n=10, 
        d=2, 
        x0=None, 
        tol=1e-4,
        method="Powell",
        opt_tol=1e-5,
        x0_range=[[-1, 1],[-1, 1]],
        sample_x0=False,
        rng=None, 
        seed=None,

):
    """Estimate the minima of the potential function.

    Args:
        model (PLNN): Model instance.
        tilt (arraylike): Tilt vector.
        n (int, optional): Number of initial guesses. Defaults to 10.
            Inferred if `x0` is given explicitly.
        d (int, optional): Dimension. Defaults to 2.
            Inferred if `x0` is given explicitly.
        x0 (array, optional): Inital guess points. Shape (n,d). Defaults to None.
        tol (float, optional): Minimum Euclidean distance distinguishing 
            potential fixed points. Defaults to 1e-4.
        rng (np.Generator, optional): Random number generator. Defaults to None.
        seed (int, optional): Seed. Defaults to None.

    Returns:
        np.ndarray : Unique estimated minima. Shape (?, d).
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    if x0 is None:
        x0 = np.nan * np.ones([n, d])
        if sample_x0:
            for i in range(d):
                x0[:,i] = rng.uniform(*x0_range[i], size=n)
        else:
            for i in range(d):
                x0[:,i] = np.linspace(*x0_range[i], n)
    else:
        n = x0.shape[0]
    
    f = lambda x: model.eval_phi_with_tilts(0, jnp.array(x), jnp.array(tilt))
    
    results = []
    for x in x0:
        optres = minimize(
            f, x, 
            method=method,
            tol=opt_tol,
        )
        if optres.success:
            results.append(optres.x)
    
    unique_minima = []
    for x1 in results:
        isnew = True
        for x2 in unique_minima:
            dist = np.sqrt(np.sum(np.square(x1 - x2)))
            if dist < tol:
                isnew = False
                break
        if isnew:
            unique_minima.append(x1)
    
    return np.array(unique_minima)


def find_minima_along_tilt_trajectory(
        model,
        tilt_trajectory,
        ndivisions=10,
        n=10, 
        d=2, 
        x0_range=[[-4,4],[-4,4]], 
        tol=1e-2,
        method="Powell",
        opt_tol=1e-6,
        rng=None, 
        seed=None,
):
    """Compute minima model's potential along a trajectory of tilt values.

    Args:
        model (_type_): _description_
        tilt_trajectory (_type_): _description_
        ndivisions (int, optional): _description_. Defaults to 10.
        n (int, optional): _description_. Defaults to 10.
        d (int, optional): _description_. Defaults to 2.
        x0_range (list, optional): _description_. Defaults to [[-4,4],[-4,4]].
        tol (_type_, optional): _description_. Defaults to 1e-2.
        method (str, optional): _description_. Defaults to "Powell".
        opt_tol (_type_, optional): _description_. Defaults to 1e-6.
        rng (_type_, optional): _description_. Defaults to None.
        seed (_type_, optional): _description_. Defaults to None.

    Returns:
        Array: Minima in the potential corresponding to each value of tilts.
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    k, tilt_dim = tilt_trajectory.shape
    assert np.shape(x0_range) == (d, 2), "x0_range should have shape (d, 2)"

    # Divide the trajectory into segments.
    ndivisions = min(n, ndivisions)
    div_points, div_idxs = divide_trajectory(
        tilt_trajectory, 
        npoints=ndivisions, 
    )

    # Estimate minima at each of the division points.
    div_minima_list = []
    for i in range(ndivisions):
        mins = estimate_minima(
            model, div_points[i],
            n=n, d=d, 
            tol=tol, 
            sample_x0=False,
            x0_range=x0_range,
            method=method,
            opt_tol=opt_tol,
            rng=rng
        )
        div_minima_list.append(mins)

    # Fill in the missing pieces along the trajectory
    full_traj_minima_list = []
    for i in range(ndivisions - 1):
        # p0, p1 = div_points[i], div_points[i+1]
        idx0, idx1 = div_idxs[i], div_idxs[i+1]
        mins0, mins1 = div_minima_list[i], div_minima_list[i+1]
        mins_ests = np.vstack([mins0, mins1])
        
        full_traj_minima_list.append(mins0)  # add mins at current point
        for tidx in range(idx0 + 1, idx1):
            tilt = tilt_trajectory[tidx]
            mins = estimate_minima(
                model, tilt,
                tol=tol, 
                x0=mins_ests, 
                method=method,
                opt_tol=opt_tol,
                rng=rng
            )
            full_traj_minima_list.append(mins)
    full_traj_minima_list.append(mins1)
    
    return full_traj_minima_list


def divide_trajectory(traj, npoints):
    """Divide a trajectory into roughly equal segments wrt arclength.

    Returns:
        (array) Points of the given trajectory dividing it into segments.
        (array) Corresponding indices of the points in the original trajectory.
    """
    # Compute distances between consecutive points in the trajectory
    p0 = np.roll(traj, 1, axis=0)
    p0[0,:] = traj[0,:]
    ds = np.sqrt(np.sum(np.square(traj - p0), axis=1))
    # Calculate the proportion of the arc from the initial point
    s_cumulative = np.cumsum(ds) / np.sum(ds)
    # Divide the trajectory into approximately equal segments wrt arclength
    s_samps = np.linspace(0, 1, npoints, endpoint=True)
    distances = np.abs(s_samps[:,None] - s_cumulative[None,:])
    samp_idxs = np.argmin(distances, axis=1)
    return traj[samp_idxs,:], samp_idxs


def check_minima_trajectories_for_bifurcations(
        tilt_trajectory,
        mins_trajectory,
):
    tilt0 = tilt_trajectory[0]
    mins0 = mins_trajectory[0]
    num_mins0 = len(mins0)
    bif_idxs = []
    bif_tilt_pairs = []
    bif_min_pairs = []
    potential_bifs = []
    for i in range(1, len(tilt_trajectory)):
        mins1 = mins_trajectory[i]
        tilt1 = tilt_trajectory[i]
        num_mins1 = len(mins1)
        if num_mins0 != num_mins1:
            # bifurcation may have occurred between start and stop
            bif_idxs.append(i-1)
            bif_tilt_pairs.append((tilt0, tilt1))
            bif_min_pairs.append((mins0, mins1))
            for m in mins0:
                potential_bifs.append(m)
            for m in mins1:
                potential_bifs.append(m)
        tilt0 = tilt1
        mins0 = mins1
        num_mins0 = num_mins1
    
    return np.array(potential_bifs), bif_idxs, bif_tilt_pairs, bif_min_pairs

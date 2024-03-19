"""Continuation algorithm to trace bifurcation curves in parameter space.

"""

import warnings
import numpy as np


def trace_curve(
        x0, 
        p0, 
        F, 
        Fx, 
        dxFxPhi, 
        Fp, 
        maxiter=10000,
        ds=1e-2,
        rho=1e-1,
        min_ds=1e-6,
        max_ds=1e-1,
        max_delta_p=1e-2,
        plims=None,
        newton_tol=1e-5,
        rng=None,
        seed=None,
        verbosity=0,
):
    """Implements a Pseudo-Arclength Continuation Algorithm.

    Locate and trace steady bifurcations, given an initial guess of a stable
    fixed point, and functions defining the dynamical system.

    [*] Refer to Sections 5.1-5.3 of Methods of Nonlinear Analysis, H. Riecke.

    Args:
        x0 (ndarray) : initial state vector.
        p0 (ndarray) : initial parameter vector.
        F (callable) : ODE system equation. Returns ndarray of shape (dimx,).
        Fx (callable) : System Jacobian. Returns ndarray of shape (dimx, dimx).
        dxFxPhi (callable) : Dx[Fx @ Phi]. Returns ndarray of shape (dimx, dimx).
        Fp (callable) : Derivative of F with respect to parameters P. 
            Returns ndarray of shape (dimx, dimp).
        maxiter (int) : maximum number of steps in continuation algorithm. 
            Default 10000.
        ds (float) : initial step size. Default 1e-2.
        min_ds (float) : minimum arclength step size. Default 1e-6.
        max_ds (float) : maximum arclength step size. Default 1e-1.
        max_delta_p (float) : maximum change allowed for parameter vector p. Default 1e-2.
        rho (float) : small parameter used for estimation of z when Fx is noninvertible.
        plims (2d list) : limits on each parameter. Continuation halts when limits reached.
        verbosity (int) : degree of verbosity.
    Returns:
        xs : ndarray - state space points on bifurcation curves. 
        ps : ndarray - parameter points on bifurcation curves.
        info : dict - dictionary containing any miscellaneous saved information.
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    if verbosity > 0:
        print(f"Tracing curve:")
        print(f"  x0: {x0}")
        print(f"  p0: {p0}")
        print(f"  ds: {ds:.2g}")
        print(f"  min_ds: {min_ds:.2g}")
        print(f"  max_ds: {max_ds:.2g}")
        print(f"  max_delta_p: {max_delta_p:.2g}")
        print(f"  rho: {rho}")

    dimx = len(x0)  # dimension of state space
    dimp = len(p0)  # dimension of parameter space

    info = {}  # dictionary for informative output
    info['eigs'] = []
    info['dets'] = []
    info['failed_to_converge_ps'] = []
    info['failed_to_converge_xs'] = []
    info['failed_to_converge_flags'] = []
    info['failed_to_converge_reasons'] = []
    info['critical_ps'] = []
    info['critical_xs'] = []
    info['critical_dets'] = []
    info['tauxs'] = []
    info['taups'] = []
    info['x0_est'] = np.array(x0)
    info['p0_est'] = np.array(p0)

    # Determine initial approximation of state space tangent vector phi.
    # Enforce a normalization condition, stipulating the first component of 
    # phi is equal to 1. Refer to pg 60 eq 37 of [*].
    e1 = np.zeros(dimx)
    e1[0] = 1  # e1 is 1st standard basis vector.
    L = np.array(Fx(x0))
    L[0,:] = e1  # replace 1st equation of Fx with normalizing condition.
    phi0 = np.linalg.solve(L, e1)

    # Initialize the extended tangent vector w=(x', 0, p1', p2')
    w = np.zeros(2*dimx + dimp, dtype=np.float64)
    w[0:dimx] = phi0
    w[dimx:2*dimx] = 0
    w[2*dimx:] = 1
    w /= np.linalg.norm(w)

    # Initialize normalization index kidx and normalization vector ek
    kidx = 0
    ek = np.zeros(dimx, dtype=np.float64)
    ek[kidx] = 1.

    # Initialize matrix M, the LHS of the equation in step 2 on pg 58 of [*]. 
    M = np.zeros([2*dimx + dimp, 2*dimx + dimp])
    M[0:dimx,2*dimx:] = Fp(x0, p0)  # upper right block is d/dp[F]
    
    # Take an initial step to find the fold curve
    M[-2,dimx:2*dimx] = ek
    M[-1,0:dimx] = w[0:dimx]
    M[-1,-dimp:] = w[-dimp:]
    stepper = get_stepper(
        x0, phi0, p0, w[0:dimx], w[dimx:2*dimx], w[-dimp:], ds, 
        M, F, Fx, dxFxPhi, dimx, dimp, ek, verbosity
    )

    x1, phi1, p1, converged = newton(
        x0, phi0, p0, stepper, 
        tol=newton_tol
    )
    if not converged:
        msg = f"Initial step with x0={x0}, p0={p0} did not converge."
        warnings.warn(msg)
        return [], [], info
    
    w_init = update_w(w, x0, x1, p0, p1, dimx, dimp)
    x_init = x1
    phi_init = phi1
    p_init = p1
    
    info['x0'] = x_init
    info['phi0'] = phi_init
    info['p0'] = p_init
    
    path_xs_fwd = [x_init]
    path_ps_fwd = [p_init]
    path_xs_rev = [x_init]
    path_ps_rev = [p_init]
    ds_init = ds
    for direction in [1, -1]:
        s0 = 0
        ds = ds_init
        x0 = x_init.copy()
        phi0 = phi_init.copy()
        p0 = p_init.copy()
        w = w_init.copy() * direction
        ek = np.zeros(dimx, dtype=np.float64)
        ek[kidx] = 1.
        ek, kidx = update_normalization_vector(phi0, ek, kidx)
        xs = path_xs_fwd if direction == 1 else path_xs_rev
        ps = path_ps_fwd if direction == 1 else path_ps_rev
        ds_adj_flag = 0
        # Main loop
        for i in range(maxiter):
            wx = w[0:dimx]
            wphi = w[dimx:2*dimx]
            wp = w[-dimp:]
            M[-2,dimx:2*dimx] = ek
            M[-1,0:dimx] = wx
            M[-1,-dimp:] = wp

            deltap = np.inf
            stepped = False
            step_attempts = 0
            break_flag = 0
            while not stepped:
                s1 = s0 + ds
                stepper = get_stepper(
                    x0, phi0, p0, wx, wphi, wp, ds, 
                    M, F, Fx, dxFxPhi, dimx, dimp, ek, verbosity
                )

                x1, phi1, p1, converged = newton(
                    x0, phi0, p0, stepper, 
                    tol=newton_tol
                )
                
                if converged:
                    # Newton iteration converged, so check the change in p.
                    # If the change exceeds the allowable maximum change,
                    # reduce ds if possible, regardless of prior value.
                    deltap = np.linalg.norm(p1 - p0)
                    if deltap > max_delta_p and i > 0:
                        if np.abs(ds / 2) >= min_ds:
                            ds, ds_adj_flag = decrease_ds(ds, 0, verbosity)
                        else:
                            break_flag = 1
                            break_reason = "ds < min_ds and delta_p > max_delta_p"
                            break
                    else:
                        if np.abs(ds * 2) <= max_ds:
                            ds, ds_adj_flag = increase_ds(ds, ds_adj_flag, verbosity)
                        stepped = True
                else:
                    # Newton iteration failed to converge. Reduce ds if possible,
                    # regardles of any prior decrease or increase.
                    if np.abs(ds / 2) >= min_ds:
                        ds, ds_adj_flag = decrease_ds(ds, 0, verbosity)
                    else:
                        break_flag = 2
                        break_reason = "Newton failed to converge and ds < min_ds."
                        break
                step_attempts += 1

            w = update_w(w, x0, x1, p0, p1, dimx, dimp)
            s0 = s1
            x0 = x1
            phi0 = phi1
            p0 = p1
            ek, kidx = update_normalization_vector(phi0, ek, kidx)
                    
            if break_flag:
                if break_flag == 1 or break_flag == 2:
                    info['failed_to_converge_ps'].append(p1)
                    info['failed_to_converge_xs'].append(x1)
                    info['failed_to_converge_flags'].append(break_flag)
                    info['failed_to_converge_reasons'].append(break_reason)
                if verbosity > 0:
                    print(f"  Halted while attempting to step: {break_reason}")
                break

            if plims is not None:
                reached_boundary = False
                for didx in range(len(p0)):
                    if p0[didx] < plims[didx][0] or p0[didx] > plims[didx][1]:
                        if verbosity > 0:
                            msg = f"Reached p{didx} boundary at iter {i}. "
                            msg += f"p={p0}"
                            print(msg)
                        reached_boundary = True
                        break
                if reached_boundary:
                    break

            xs.append(x1)
            ps.append(p1)
    
    # Concatenate forward and reverse paths into a single path.
    xs = np.concatenate([np.flip(path_xs_rev, axis=0)[:-1], path_xs_fwd])
    ps = np.concatenate([np.flip(path_ps_rev, axis=0)[:-1], path_ps_fwd])
    info['xs_path_fwd'] = np.array(path_xs_fwd)
    info['ps_path_fwd'] = np.array(path_ps_fwd)
    info['xs_path_rev'] = np.array(path_xs_rev)
    info['ps_path_rev'] = np.array(path_ps_rev)
    return xs, ps, info


def newton(x0, phi0, p0, step_func, tol=1e-5, maxiter=10000):
    """Implements Newton's method.
    Args:
        x0 : initial guess for x.
        phi0 : initial guess for phi.
        p0 : initial guess for p.
        step_func : function returning steps dx, dphi, dp.
        tol : float - tolerance. Halts when full vector <x,phi,p> change 
            is less than tol.
        maxiter : maximum number of steps.
    Returns:
        x1 : ndarray - new value of x1.
        phi1 : ndarray - new value of phi1.
        p1 : ndarray - new value of p1.
        converged : bool - whether the algorithm converged or not.
    """
    diff = np.inf
    converged = False
    for i in range(maxiter):
        dxp = step_func(x0, phi0, p0)
        dx = dxp[0:2]
        dphi = dxp[2:4]
        dp = dxp[4:]
        x1 = x0 + dx
        phi1 = phi0 + dphi
        p1 = p0 + dp
        diff = np.linalg.norm(dxp)
        x0 = x1
        phi0 = phi1
        p0 = p1
        if diff < tol:
            converged = True
            break       
    return x1, phi1, p1, converged


def decrease_ds(ds, ds_adj_flag, verbosity=0):
    """Decrease ds based on prior adjustment.
    
    Args:
        ds (float) : Current value of ds.
        ds_adj_flag (int) : -1, 1, or 0, based on whether ds was decreased, 
            increased, or unchanged, in the prior step.
    Returns:
        (float) : updated value of ds
        (int) : -1, 1, or 0, if ds was decreased, increased, or unchanged.
    """
    if ds_adj_flag == 1:  
        # ds was just increased, so wait to reduce
        return ds, 0
    else:
        ds /= 2
        if verbosity > 2: print(f"\t\tDecreased ds to {ds:.3g}")
        return ds, -1


def increase_ds(ds, ds_adj_flag, verbosity=0):
    """Increase ds based on prior adjustment.
    
    Args:
        ds (float) : Current value of ds.
        ds_adj_flag (int) : -1, 1, or 0, based on whether ds was decreased, 
            increased, or unchanged, in the prior step.
    Returns:
        (float) : updated value of ds
        (int) : -1, 1, or 0, if ds was decreased, increased, or unchanged.
    """
    if ds_adj_flag == -1:  
        # ds was just reduced, so wait to increase
        return ds, 0
    else:
        ds *= 2
        if verbosity > 2: print(f"\t\tIncreased ds to {ds:.3g}")
        return ds, 1
    

def get_stepper(
        x0, phi0, p0,
        wx, wphi, wp,
        ds,
        M, 
        F, 
        Fx, 
        dxFxPhi, 
        dimx, 
        dimp, 
        ek, 
        verbosity,
):
    def stepper(x, phi, p):
        fx = Fx(x)
        M[0:dimx, 0:dimx] = fx
        M[dimx:2*dimx, dimx:2*dimx] = fx
        M[dimx:2*dimx, 0:dimx] = dxFxPhi(x, phi)
        RHS = -np.concatenate([
            F(x, p), 
            fx@phi, 
            [ek@phi-1], 
            [wx@(x-x0) + wphi@(phi-phi0) + wp@(p-p0) - ds]
        ])
        try:
            dxp = np.linalg.solve(M, RHS)
        except np.linalg.LinAlgError as e:
            if verbosity > 2:
                print(f"\t\tCaught error: {e}\n\t\t\tComputing lstsq")
            dxp = np.linalg.lstsq(M, RHS, rcond=None)[0]
        return dxp
    return stepper


def update_w(w, x0, x1, p0, p1, dimx, dimp):
    w[0:dimx] = x1 - x0
    w[dimx:2*dimx] = 0.
    w[-dimp:] = p1 - p0
    w /= np.linalg.norm(w)
    return w


def update_normalization_vector(phi0, ek, kidx):
    if np.abs(phi0[kidx]) < 0.5 * np.max(np.abs(phi0)):
        ek[kidx] = 0.
        kidx = np.argmax(np.abs(phi0))
        ek[kidx] = 1.
    return ek, kidx
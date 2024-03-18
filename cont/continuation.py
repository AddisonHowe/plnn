"""Continuation algorithm to trace bifurcation curves in parameter space.

"""

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
        random_p_increment=False,
        p_increment_value=None,
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
        d : dict - dictionary containing any miscellaneous saved information.
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

    d = {}  # dictionary for informative output
    d['eigs'] = []
    d['dets'] = []
    d['failed_to_converge_ps'] = []
    d['failed_to_converge_xs'] = []
    d['failed_to_converge_flags'] = []
    d['failed_to_converge_reasons'] = []
    d['critical_ps'] = []
    d['critical_xs'] = []
    d['critical_dets'] = []
    d['tauxs'] = []
    d['taups'] = []

    # Determine initial approximation of state space tangent vector phi.
    # Enforce a normalization condition, stipulating the first component of 
    # phi is equal to 1. Refer to pg 60 eq 37 of [*].
    e1 = np.zeros(dimx)
    e1[0] = 1  # e1 is 1st standard basis vector.
    L = np.array(Fx(x0))
    L[0,:] = e1  # replace 1st equation of Fx with normalizing condition.
    phi0 = np.linalg.solve(L, e1)

    # Initialize matrix M, the LHS of the equation in step 2 on pg 58 of [*]. 
    M = np.zeros([2*dimx + dimp, 2*dimx + dimp])
    M[0:dimx,2*dimx:] = Fp(x0, p0)  # upper right block is d/dp[F]

    # Initialize tracking arrays
    xs = [x0]
    ps = [p0]
    critical_ps = []

    # Main algorithm: Refer to top of pg 59 of [*].
    tau0 = None
    det0 = None
    s0 = 0  # initialize arclength parameter
    ds_adj_flag = 0

    if random_p_increment:
        p_increment = 2*rng.rand(dimp) - 1
        p_increment = p_increment / np.linalg.norm(p_increment)
    elif p_increment_value is not None:
        p_increment = np.array(p_increment_value) / np.sqrt(
            np.sum(np.square(p_increment_value)))
    else:
        p_increment = np.ones(dimp) / np.sqrt(dimp)

    for i in range(maxiter):           
        
        # Compute the tangent vector tau by solving d/dx[F] @ z = -d/dp[F].
        # Here we assume that each parameter is a linear component of the dynamics.
        fx0 = Fx(x0)
        try:
            # Try solving d/dx[F] @ z = -d/dp[F] @ d/ds[p], assuming 
            # non-singular d/dx[F].
            z = np.linalg.solve(fx0, -Fp(x0, p0)@p_increment)
        except np.linalg.LinAlgError as e:
            # Catch error in the case of singular d/dx[F].
            if verbosity > 0:
                print("At singular point")
            A = np.linalg.inv(fx0 - rho*np.eye(dimx))
            z = np.linalg.lstsq(fx0, -Fp(x0, p0)@p_increment, rcond=None)[0]
            ztol = 1e-8
            zdiff = np.inf
            counter = 0
            maxiterz = 10000
            while zdiff > ztol and counter < maxiterz:
                z1 = A@z
                zdiff = np.linalg.norm(z1-z)
                z = z1
                counter += 1
            if counter == maxiterz:
                if verbosity > 0:
                    print("Failed to converge for singular z computation!")
        
        tau1 =  np.concatenate([1/np.sqrt(np.dot(z,z) + 1.) * z, p_increment])
        tau1 /= np.linalg.norm(tau1)
        
        if tau0 is not None and tau0 @ tau1 < 0:
            tau1 *= -1
        taux = tau1[0:dimx]
        taup = tau1[dimx:]
        tau0 = tau1
        d['tauxs'].append(taux)
        d['taups'].append(taup)
        
        M[-1,0:dimx] = taux
        M[-1,-dimp:] = taup

        deltap = np.inf
        stepped = False
        step_attempts = 0
        break_flag = 0
        while not stepped:
            
            s1 = s0 + ds  # Increment arclength parameter

            # Define the stepper function. This evaluates the given 
            def stepper(x, phi, p):
                fx = Fx(x)
                M[0:dimx, 0:dimx] = fx
                M[dimx:2*dimx, dimx:2*dimx] = fx
                M[dimx:2*dimx, 0:dimx] = dxFxPhi(x, phi)
                M[2*dimx, dimx:2*dimx] = 2*phi
                RHS = -np.concatenate([
                    F(x, p), 
                    fx@phi, 
                    [phi@phi-1], 
                    [taux@(x-x0) + taup@(p-p0) - ds]
                ])
                try:
                    dxp = np.linalg.solve(M, RHS)
                except np.linalg.LinAlgError as e:
                    if verbosity > 2:
                        print(f"\t\tCaught error: {e}\n\t\t\tComputing lstsq")
                    dxp = np.linalg.lstsq(M, RHS, rcond=None)[0]
                return dxp

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

        s0 = s1

        det1 = np.linalg.det(M)
        detfx0 = np.linalg.det(Fx(x0))
        detfx1 = np.linalg.det(Fx(x1))
        # print(det1, detfx0, detfx1)
        d['dets'].append(det1)
        if det0 is not None and det0 * det1 < 0:
            critical_ps.append((p0 + p1) / 2)
            d['critical_ps'].append((p0 + p1) / 2)
            d['critical_xs'].append((x0 + x1) / 2)
            d['critical_dets'].append(det1)
            break_flag = 3
            break_reason = "Determinant changed sign at critical point."
            if verbosity > 2:
                print("\t\tDeterminant of jacobian changed signs!!!")

        # p_increment = p1 - p0
        # p_increment /= np.linalg.norm(p_increment) 
        x0 = x1
        phi0 = phi1
        p0 = p1
        det0 = det1
                
        if break_flag:
            if break_flag == 1 or break_flag == 2:
                d['failed_to_converge_ps'].append(p1)
                d['failed_to_converge_xs'].append(x1)
                d['failed_to_converge_flags'].append(break_flag)
                d['failed_to_converge_reasons'].append(break_reason)
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
    
    return np.array(xs), np.array(ps), np.array(critical_ps), d


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
    
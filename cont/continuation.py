import numpy as np

"""
Implements a continuation algorithm to trace bifurcation curves in parameter space.
"""

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


def trace_curve(x0, p0, F, Fx, dxFxPhi, Fp, **kwargs):
    """Implements a Pseudo-Arclength Continuation Algorithm.

    [*] Refer to Sections 5.1-5.3 of Methods of Nonlinear Analysis, H. Riecke.

    Args:
        x0 : ndarray - initial state vector.
        p0 : ndarray - initial parameter vector.
        F : callable - ODE system equation. Returns ndarray of shape (dimx,).
        Fx : callable - System Jacobian. Returns ndarray of shape (dimx, dimx).
        dxFxPhi : callable - Dx[Fx @ Phi]. Returns ndarray of shape (dimx, dimx).
        Fp : callable - Derivative of F with respect to parameters P. 
            Returns ndarray of shape (dimx, dimp)
    Optional Args:
        verbosity : int - degree of verbosity.
        maxiter : int - maximum number of steps in continuation algorithm. Default 10000.
        ds : float - initial step size. Default 1e-2.
        min_ds : float - minimum arclength step size. Default 1e-6.
        max_ds : float - maximum arclength step size. Default 1e-1.
        max_delta_p : float - maximum change allowed for parameter vector p. Default 1e-2.
        rho : float - small parameter used for estimation of z when Fx is noninvertible.
        plims : 2d list - limits on each parameter. Continuation halts when limits reached.
    Returns:
        xs : ndarray - state space points on bifurcation curves. 
        ps : ndarray - parameter points on bifurcation curves.
        d : dict - dictionary containing any miscellaneous saved information.
    """

    # Process keyword args
    maxiter = kwargs.get('maxiter', 10000)
    ds = kwargs.get('ds', 1e-2)
    rho = kwargs.get('rho', 1e-1)
    verbosity = kwargs.get('verbosity', 0)
    plims = kwargs.get('plims', None)
    min_ds = kwargs.get('min_ds', 1e-6)
    max_ds = kwargs.get('max_ds', 1e-1)
    max_delta_p = kwargs.get('max_delta_p', 1e-2)

    if verbosity > 0:
        print(f"x0={x0}")
        print(f"p0={p0}")
        print(f"ds={ds:.2e}")
        print(f"min_ds={min_ds:.2g}")
        print(f"max_ds={max_ds:.2g}")
        print(f"max_delta_p={max_delta_p:.2g}")
        print(f"rho={rho}")

    dimx = len(x0)  # dimension of state space
    dimp = len(p0)  # dimension of parameter space

    d = {}  # dictionary for informative output
    d['eigs'] = []
    d['dets'] = []
    d['failed_to_converge_ps'] = []
    d['failed_to_converge_xs'] = []

    # Determine initial approximation of state space tangent vector phi.
    e1 = np.zeros(dimx)
    e1[0] = 1
    L = Fx(x0)
    L[0,:] = e1
    phi0 = np.linalg.solve(L, e1)

    # Construct M matrix: See Pg. 58 of [*].
    M = np.zeros([2*dimx + 2, 2*dimx + 2])
    M[0:dimx,2*dimx:] = Fp(x0, p0)


    y = np.zeros(2*dimx + dimp)
    y[:] = np.concatenate([x0, phi0, p0])

    xs = [x0]
    ps = [p0]
    critical_ps = []

    tau0 = None
    det0 = None
    s0 = 0
    for i in range(maxiter):
        fx0 = Fx(x0)
        try:
            z = np.linalg.solve(fx0, -np.sum(Fp(x0, p0), axis=1))
        except np.linalg.LinAlgError as e:
            if verbosity > 0:
                print("At Fold point")
            A = np.linalg.inv(fx0 - rho*np.eye(dimx))
            z = np.linalg.lstsq(fx0, -np.sum(Fp(x0, p0), axis=1), rcond=None)[0]
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
                    print("Failed to converge for Fold Point z computation!")
        
        tau1 = 1/np.sqrt(np.dot(z,z) + 2) * np.concatenate([z, [1, 1]])
        
        if tau0 is not None and tau0@tau1 < 0:
            tau1 *= -1
        taux = tau1[0:dimx]
        taup = tau1[dimx:]
        tau0 = tau1
        
        M[-1,0:dimx] = taux
        M[-1,-dimp:] = taup

        deltap = np.inf
        deltap_exceeded = False
        stepped = False
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
                    if verbosity > 0:
                        print(e)
                    dxp = np.linalg.lstsq(M, RHS, rcond=None)[0]
                return dxp

            x1, phi1, p1, converged = newton(x0, phi0, p0, stepper)     
            
            deltap = np.linalg.norm(p1 - p0)
            if deltap > max_delta_p:
                if np.abs(ds / 2) > min_ds:
                    ds /= 2
                else:
                    deltap_exceeded = True
                    break
            else:
                stepped = True
                if np.abs(ds * 2) < max_ds:
                    ds *= 2

        s0 = s1

        det1 = np.linalg.det(M)
        eigs1 = np.linalg.eig(M[0:dimx,0:dimx])[0]
        d['eigs'].append(eigs1)
        d['dets'].append(det1)
        if det0 is not None and det0 * det1 < 0:
            critical_ps.append((p0 + p1)/2)

        x0 = x1
        phi0 = phi1
        p0 = p1
        det0 = det1

        if not converged:
            d['failed_to_converge_ps'].append(p1)
            d['failed_to_converge_xs'].append(x1)
            if verbosity > 0:
                print(f"Newton failed to converge in {maxiter} iterations!")
            break

        if deltap_exceeded:
            if verbosity > 0:
                msg = f"Delta p = {deltap:.3g} > {max_delta_p}, "
                msg += f"but ds reached minimum of {min_ds}."
                print(msg)
            break

        if plims is not None:
            if p0[0] < plims[0][0] or p0[0] > plims[0][1]:
                if verbosity > 0:
                    msg = f"Reached p1 boundary at iter {i}. "
                    msg += f"p=({p0[0]:.3g}, {p0[1]:.3g})"
                    print(msg)
                break
            if p0[1] < plims[1][0] or p0[1] > plims[1][1]:
                if verbosity > 0:
                    msg = f"Reached p2 boundary at iter {i}. "
                    msg += f"p=({p0[0]:.3g}, {p0[1]:.3g})"
                    print(msg)
                break

        xs.append(x1)
        ps.append(p1)
    
    return np.array(xs), np.array(ps), critical_ps, d

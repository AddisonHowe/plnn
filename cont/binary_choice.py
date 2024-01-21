import numpy as np
import matplotlib.pyplot as plt
from cont.continuation import trace_curve

F = lambda x, p: -np.array([
        4*x[0]**3 - 8*x[0]*x[1] - p[0], 
        4*x[1]**3 + 3*x[1]*x[1] - 4*x[0]*x[0] + 2*x[1] + p[1],
    ])

Fx = lambda x: -np.array([
        [12*x[0]**2 - 8*x[1],   -8*x[0]],
        [-8*x[0],               12*x[1]**2 + 6*x[1] + 2],
    ])

dxFxPhi = lambda x, phi: np.array([
        [-24*x[0]*phi[0] + 8*phi[1],    8*phi[0]],
        [8*phi[0],                      -(24*x[1] + 6)*phi[1]],
    ])

Fp = lambda x, p : np.array([[1, 0], [0, -1]])


XSTARTS = [
    [[-0.9899,  0.5086], 'g'],
    [[ 1.,  0.5], 'brown'],
    [[ 0.1891, -0.1581], 'b'],
    [[-0.1891, -0.1581], 'b'],
    [[-0.961,  0.54], 'g'],
    [[ 0.961,  0.54], 'brown'],
    [[ 0.,  0.01], 'b'],
]

P1 = lambda x: 4*x[0]**3 - 8*x[0]*x[1]
P2 = lambda x: -4*x[1]**3 - 3*x[1]**2 + 4*x[0]**2 - 2*x[1]

maxiter = 10000
ds = 1e-3
min_ds = 1e-8
max_ds = 1e-1
max_delta_p = 1e-1
rho = 1e-1
P1LIMS = [-2, 2]
P2LIMS = [-1, 3]
P1_VIEW_LIMS = [-2, 2]
P2_VIEW_LIMS = [-1, 3]

def get_binary_choice_curves(p1lims=P1LIMS, p2lims=P2LIMS, xstarts=XSTARTS):
    p1lims = p1lims.copy()
    p2lims = p2lims.copy()
    p1lims[0] = min(p1lims[0], P1LIMS[0])
    p1lims[1] = max(p1lims[1], P1LIMS[1])
    p2lims[0] = min(p2lims[0], P2LIMS[0])
    p2lims[1] = max(p2lims[1], P2LIMS[1])

    curves_p = []
    colors = []
    for i in range(len(xstarts)):
        x0 = np.array(xstarts[i][0])
        col = xstarts[i][1]
        p0 = np.array([P1(x0), P2(x0)])
        for sign in [1, -1]:
            _, ps, _, _ = trace_curve(
                x0, p0, F, Fx, dxFxPhi, Fp,
                maxiter=maxiter, 
                ds=ds*sign,
                min_ds=min_ds,
                max_ds=max_ds,
                max_delta_p=max_delta_p,
                rho=rho,
                plims=[p1lims, p2lims],
                verbosity=0,
            )
            curves_p.append(ps)
            colors.append(col)
    return curves_p, colors

def main():
    fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    curves_x = []
    curves_p = []
    crit_ps = []
    eigs = []
    for i in range(len(XSTARTS)):
        x0 = np.array(XSTARTS[i][0])
        col = XSTARTS[i][1]
        p0 = np.array([P1(x0), P2(x0)])
        for sign in [1, -1]:
            xs, ps, cps, d = trace_curve(
                x0, p0, F, Fx, dxFxPhi, Fp,
                maxiter=maxiter, 
                ds=ds*sign,
                min_ds=min_ds,
                max_ds=max_ds,
                max_delta_p=max_delta_p,
                rho=rho,
                plims=[P1LIMS, P2LIMS],
                verbosity=1,
            )
            curves_x.append(xs)
            curves_p.append(ps)
            crit_ps.append(cps)
            eigs.append(np.array(d['eigs']))

            # ax1.plot(ps[0,0], ps[0,1], 'o', alpha=0.2, color=col)
            # ax2.plot(xs[0,0], xs[0,1], 'o', alpha=0.2, color=col)
            if len(ps) > 1:
                ax1.plot(ps[1,0], ps[1,1], '*', alpha=0.6, color=col)
                ax1.plot(ps[:,0], ps[:,1], '-', alpha=1, color=col)

                ax2.plot(xs[1,0], xs[1,1], '*', alpha=0.6, color=col)
                ax2.plot(xs[:,0], xs[:,1], '-', alpha=1, color=col)

    ax1.set_xlim(*P1_VIEW_LIMS)
    ax1.set_ylim(*P2_VIEW_LIMS)
    ax1.set_xlabel('$p_1$')
    ax1.set_ylabel('$p_2$')

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    plt.show()

if __name__ == "__main__":
    main()
    
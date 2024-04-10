"""Bifurcation diagram for the binary choice landscape (phi1).

"""

import sys
import numpy as np

from cont.fold_curves import get_fold_curves, get_argparser, plot_diagrams


F = lambda x, p: -np.array([
        4*x[0]**3 - 8*x[0]*x[1] - p[0], 
        4*x[1]**3 + 3*x[1]*x[1] - 4*x[0]*x[0] + 2*x[1] + p[1],
    ])

J = lambda x: -np.array([
        [12*x[0]**2 - 8*x[1],   -8*x[0]],
        [-8*x[0],               12*x[1]**2 + 6*x[1] + 2],
    ])

H = lambda x: -np.array([
    [[24*x[0], -8.],
     [-8., 0.]],
    [[-8., 0.],
     [0., 24*x[1]+6.]]
])

dxFxPhi = lambda x, phi: np.array([
        [-24*x[0]*phi[0] + 8*phi[1],    8*phi[0]],
        [8*phi[0],                      -(24*x[1] + 6)*phi[1]],
    ])

Fp = lambda x, p : np.array([[1, 0], [0, -1]])


XSTARTS = [
    [[ 0.1891, -0.1581], 'b'],
    [[ 0.9201, 0.6418], 'brown'],
    [[-0.9201, 0.6418], 'g'],
]

P1 = lambda x: 4*x[0]**3 - 8*x[0]*x[1]
P2 = lambda x: -4*x[1]**3 - 3*x[1]**2 + 4*x[0]**2 - 2*x[1]

MAXITER = 10000
DS = 1e-4
MIN_DS = 1e-8
MAX_DS = 1e-2
MAX_DELTA_P = 1e-2
RHO = 1e-1
P1LIMS = [-20, 20]
P2LIMS = [-20, 20]

P1_VIEW_LIMS = [-2, 2]
P2_VIEW_LIMS = [-1, 3]
X_VIEW_LIMS = [-2, 2]
Y_VIEW_LIMS = [-2, 2]


def get_binary_choice_curves(
        p1lims=P1LIMS, 
        p2lims=P2LIMS, 
        xstarts=XSTARTS,
        rng=None,
        seed=None,
):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    p1lims = p1lims.copy()
    p2lims = p2lims.copy()
    p1lims[0] = min(p1lims[0], P1LIMS[0])
    p1lims[1] = max(p1lims[1], P1LIMS[1])
    p2lims[0] = min(p2lims[0], P2LIMS[0])
    p2lims[1] = max(p2lims[1], P2LIMS[1])

    curves_p, colors = get_fold_curves(
        F, J, dxFxPhi, Fp,
        xstarts, 
        P1, 
        P2,
        p1lims=p1lims, 
        p2lims=p2lims, 
        maxiter=MAXITER, 
        ds=DS, 
        min_ds=MIN_DS, 
        max_ds=MAX_DS,
        max_delta_p=MAX_DELTA_P,
        rho=RHO,
        rng=rng,
    )
    return curves_p, colors


def main(args):
    plot_diagrams(
        args,
        XSTARTS, P1, P2, 
        F, J, dxFxPhi, Fp,
        MAXITER, DS, MIN_DS, MAX_DS, MAX_DELTA_P, RHO, P1LIMS, P2LIMS,
        P1_VIEW_LIMS, P2_VIEW_LIMS, X_VIEW_LIMS, Y_VIEW_LIMS,
    )
    

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args(sys.argv[1:])
    main(args)

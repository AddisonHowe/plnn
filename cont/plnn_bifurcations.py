"""Bifurcation diagram for trained PLNN.

"""

import os, sys
import numpy as np
import jax
import jax.numpy as jnp

from plnn.models import DeepPhiPLNN
from plnn.io import load_model_from_directory
from cont.fold_curves import get_fold_curves, get_argparser, plot_diagrams


MAXITER = 10000
DS = 1e-3
MIN_DS = 1e-8
MAX_DS = 1e-2
MAX_DELTA_P = 1e-1
RHO = 1e-1
P1LIMS = [-4, 4]
P2LIMS = [-4, 4]

GRAD_TOL = 1e-4
XLIMS = [-4, 4]
YLIMS = [-4, 4]

P1_VIEW_LIMS = [-4, 4]
P2_VIEW_LIMS = [-4, 4]
X_VIEW_LIMS = [-4, 4]
Y_VIEW_LIMS = [-4, 4]


def get_functions_from_model(model):
    tilt_nn_w = model.get_parameters()['tilt.w'][0]
    tilt_nn_b = model.get_parameters()['tilt.b'][0]

    @jax.jit
    def F(x, p): 
        return -(model.eval_grad_phi(0., x) + p)

    @jax.jit
    def J(x): 
        return -jax.jacrev(model.eval_grad_phi, 1)(0., x)

    @jax.jit
    def dxFxPhi(x, phi):
        jphi = lambda x1, phi1: J(x1) @ phi1
        return jax.jacrev(jphi, 0)(x, phi)

    @jax.jit
    def Fp(x, p):
        return -np.eye(*tilt_nn_w.shape)

    @jax.jit
    def solve_p(x):
        return -model.eval_grad_phi(0., x)
    
    p1func = lambda x: solve_p(x)[0]
    p2func = lambda x: solve_p(x)[1]

    return F, J, dxFxPhi, Fp, solve_p, p1func, p2func


def get_xstarts(
        xstarts0, num_starts, F, p_func, 
        xlims=[[-2,2],[-2,2]], grad_tol=1e-2, color='k', 
        rng=None, seed=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)

    xstarts = []
    for x in xstarts0:
        xstarts.append(x)
    while len(xstarts) < num_starts:
        x = [rng.uniform(xlims[i][0], xlims[i][1]) for i in range(len(xlims))]
        x_tmp = np.array(x)
        if np.linalg.norm(F(x_tmp, p_func(x_tmp))) < grad_tol:
            xstarts.append([x_tmp, color])
    return xstarts


def get_plnn_bifurcation_curves(
        model, 
        num_starts=10,
        p1lims=P1LIMS, 
        p2lims=P2LIMS, 
        xstarts=[],
        xlims=XLIMS,
        ylims=YLIMS,
        color='k',
        maxiter=MAXITER,
        ds=DS,
        min_ds=MIN_DS,
        max_ds=MAX_DS,
        max_delta_p=MAX_DELTA_P,
        rho=RHO,
        rng=None,
        seed=None,
        verbosity=0
):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    p1lims = p1lims.copy()
    p2lims = p2lims.copy()
    p1lims[0] = min(p1lims[0], P1LIMS[0])
    p1lims[1] = max(p1lims[1], P1LIMS[1])
    p2lims[0] = min(p2lims[0], P2LIMS[0])
    p2lims[1] = max(p2lims[1], P2LIMS[1])

    F, J, dxFxPhi, Fp, p_func, p1func, p2func = get_functions_from_model(model)

    xstarts = get_xstarts(
        xstarts, num_starts, F, p_func, 
        xlims=[xlims, ylims], grad_tol=GRAD_TOL, color=color, 
        rng=rng
    )

    curves_p, colors = get_fold_curves(
        F, J, dxFxPhi, Fp,
        xstarts, 
        p1func, 
        p2func, 
        p1lims=p1lims, 
        p2lims=p2lims, 
        maxiter=maxiter, 
        ds=ds, 
        min_ds=min_ds, 
        max_ds=max_ds,
        max_delta_p=max_delta_p,
        rho=rho,
        verbosity=verbosity,
        rng=rng,
    )
    return curves_p, colors 


def add_args(parser):
    parser.add_argument('--model_fpath', type=str, default=None)
    parser.add_argument('--modeldir', type=str, default=None)
    parser.add_argument('-n', '--num_starts', type=int, default=100, 
                        help="number of starting values of p0 to use.")
    parser.add_argument('--progress_bar', action="store_true")
    parser.add_argument('--savedata', action="store_true")
    

def main(args):

    model_fpath = args.model_fpath
    modeldir = args.modeldir
    num_starts = args.num_starts
    savedata = args.savedata
    outdir = args.outdir
    nprng = np.random.default_rng(seed=args.seed)

    if model_fpath:
        model, _ = DeepPhiPLNN.load(model_fpath, dtype=jnp.float64)
    else:
        model = load_model_from_directory(
            modeldir, model_class=DeepPhiPLNN, dtype=jnp.float64
        )[0]
    
    F, J, dxFxPhi, Fp, p_func, p1func, p2func = get_functions_from_model(model)
    
    xstarts = get_xstarts(
        xstarts, num_starts, F, p_func, 
        xlims=[XLIMS, YLIMS], grad_tol=GRAD_TOL, color='k', 
        rng=nprng
    )

    plot_diagrams(
        vars(args),
        xstarts, p1func, p2func, 
        F, J, dxFxPhi, Fp,
        MAXITER, DS, MIN_DS, MAX_DS, MAX_DELTA_P, RHO, P1LIMS, P2LIMS,
        P1_VIEW_LIMS, P2_VIEW_LIMS, X_VIEW_LIMS, Y_VIEW_LIMS,
    )

    if savedata:
        # os.makedirs(outdir, exist_ok=True)
        pass


if __name__ == "__main__":
    parser = get_argparser()
    add_args(parser)
    args = parser.parse_args(sys.argv[1:])
    main(args)

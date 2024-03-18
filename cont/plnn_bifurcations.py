"""Bifurcation diagram for trained PLNN.

"""

import os, sys
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from plnn.models import DeepPhiPLNN
from plnn.io import load_model_from_directory
from cont.continuation import trace_curve


XSTARTS = []
maxiter = 10000
ds = 1e-3
min_ds = 1e-8
max_ds = 1e-1
max_delta_p = 1e-1
rho = 1e-1
P1LIMS = [-4, 4]
P2LIMS = [-4, 4]
P1_VIEW_LIMS = [-4, 4]
P2_VIEW_LIMS = [-4, 4]


def get_plnn_bifurcation_curves(
        model, 
        num_starts=10,
        p1lims=P1LIMS, p2lims=P2LIMS, xstarts=[],
        random_p_increment=False,
        p_increment_value=None,
        color='k',
        rng=None, seed=None,
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

    curves_p = []
    colors = []

    tilt_nn_w = model.get_parameters()['tilt.w'][0]
    tilt_nn_b = model.get_parameters()['tilt.b'][0]

    if tilt_nn_b is None:
        pass
    else:
        raise RuntimeError()

    @jax.jit
    def F(x, p): 
        # return -(model.eval_grad_phi(0., x) + tilt_nn_w @ p)
        return -(model.eval_grad_phi(0., x) + p)

    @jax.jit
    def Fx(x): 
        return -jax.jacrev(model.eval_grad_phi, 1)(0., x)

    @jax.jit
    def dxFxPhi(x, phi):
        jphi = lambda x1, phi1: Fx(x1) @ phi1
        return jax.jacrev(jphi, 0)(x, phi)

    # Fp = lambda x, p: -tilt_nn_w
    Fp = lambda x, p: -np.eye(*tilt_nn_w.shape)

    def solve_p(x):
        # return -jnp.linalg.inv(tilt_nn_w) @ model.eval_grad_phi(0., x)
        return -model.eval_grad_phi(0., x)
    
    r = 2
    grad_tol = 1e-4
    while len(xstarts) < num_starts:
        x_tmp = r * (2*rng.random(2) - 1)  # uniform in interval [-r, r]
        if np.linalg.norm(F(x_tmp, solve_p(x_tmp))) < grad_tol:
            xstarts.append([x_tmp, color])

    for i in range(len(xstarts)):        
        x0 = np.array(xstarts[i][0])
        col = xstarts[i][1]
        p0 = np.array(solve_p(x0))
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
                verbosity=verbosity,
                random_p_increment=random_p_increment,
                p_increment_value=p_increment_value,
                rng=rng,
            )
            curves_p.append(ps)
            colors.append(col)
    return curves_p, colors


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fpath', type=str, default=None)
    parser.add_argument('--modeldir', type=str, default=None)

    parser.add_argument('-n', '--num_starts', type=int, default=100, 
                        help="number of starting values of p0 to use.")
    parser.add_argument('--plot_starts', action="store_true")
    parser.add_argument('--plot_first_steps', action="store_true")
    parser.add_argument('--plot_failed_to_converge_points', action="store_true")
    parser.add_argument('--show', action="store_true")
    parser.add_argument('-s', '--saveas', type=str, 
                        default="bifcurves_plnn.pdf")
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--progress_bar', action="store_true")
    parser.add_argument('--savedata', action="store_true")
    parser.add_argument('--random_p_increment', action="store_true")
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(args)


def main(args):

    savedata = args.savedata
    outdir = args.outdir

    nprng = np.random.default_rng(seed=args.seed)

    model_fpath = args.model_fpath
    modeldir = args.modeldir
    if model_fpath:
        model, _ = DeepPhiPLNN.load(model_fpath, dtype=jnp.float64)
    else:
        model = load_model_from_directory(
            modeldir, model_class=DeepPhiPLNN, dtype=jnp.float64
        )[0]
    
    plot_starts = args.plot_starts
    plot_first_steps = args.plot_first_steps
    plot_failed_to_converge_points =args.plot_failed_to_converge_points

    tilt_nn_w = model.get_parameters()['tilt.w'][0]
    tilt_nn_b = model.get_parameters()['tilt.b'][0]
    
    if tilt_nn_b is None:
        pass
    else:
        raise RuntimeError()
        
    @jax.jit
    def F(x, p): 
        # return -(model.eval_grad_phi(0., x) + tilt_nn_w @ p)
        return -(model.eval_grad_phi(0., x) + p)

    @jax.jit
    def Fx(x): 
        return -jax.jacrev(model.eval_grad_phi, 1)(0., x)

    @jax.jit
    def dxFxPhi(x, phi):
        jphi = lambda x1, phi1: Fx(x1) @ phi1
        return jax.jacrev(jphi, 0)(x, phi)

    # Fp = lambda x, p: -tilt_nn_w
    Fp = lambda x, p: -np.eye(*tilt_nn_w.shape)


    def solve_p(x):
        # return -jnp.linalg.inv(tilt_nn_w) @ model.eval_grad_phi(0., x)
        return -model.eval_grad_phi(0., x)
    
    r = 2
    grad_tol = 1e-4
    while len(XSTARTS) < args.num_starts:
        x_tmp = r * (2*nprng.random(2) - 1)  # uniform in interval [-r, r]
        if np.linalg.norm(F(x_tmp, solve_p(x_tmp))) < grad_tol:
            XSTARTS.append([x_tmp, 'k'])
    
    fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    curves_x = []
    curves_p = []
    crit_ps = []
    eigs = []
    failed_to_converge_xs = []
    failed_to_converge_ps = []
    for i in tqdm(range(len(XSTARTS)), disable=(not args.progress_bar)):
        x0 = np.array(XSTARTS[i][0])
        col = XSTARTS[i][1]
        p0 = np.array(solve_p(x0))
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
                verbosity=args.verbosity,
                random_p_increment=args.random_p_increment,
            )
            curves_x.append(xs)
            curves_p.append(ps)
            crit_ps.append(cps)
            eigs.append(np.array(d['eigs']))
            failed_to_converge_ps.append(np.array(d['failed_to_converge_ps']))
            failed_to_converge_xs.append(np.array(d['failed_to_converge_xs']))

            if plot_starts:
                ax1.plot(ps[0,0], ps[0,1], 'o', alpha=0.2, color=col)
                ax2.plot(xs[0,0], xs[0,1], 'o', alpha=0.2, color=col)
            
            if len(ps) > 1:
                ax1.plot(ps[:,0], ps[:,1], '-', alpha=1, color=col)
                ax2.plot(xs[:,0], xs[:,1], '-', alpha=1, color=col)
                if plot_first_steps:
                    ax1.plot(ps[1,0], ps[1,1], '*', alpha=0.6, color=col)
                    ax2.plot(xs[1,0], xs[1,1], '*', alpha=0.6, color=col)
                
            if plot_failed_to_converge_points:
                for p in d['failed_to_converge_ps']:
                    ax1.plot(*p, '^', alpha=0.6, color=col)
                for x in d['failed_to_converge_xs']:
                    ax2.plot(*x, '^', alpha=0.6, color=col)

    ax1.set_xlim(*P1_VIEW_LIMS)
    ax1.set_ylim(*P2_VIEW_LIMS)
    ax1.set_xlabel('$p_1$')
    ax1.set_ylabel('$p_2$')

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    plt.tight_layout()

    if args.saveas:
        plt.savefig(args.saveas, bbox_inches='tight')

    if args.show:
        plt.show()

    if savedata:
        os.makedirs(outdir, exist_ok=True)
        np.save(f"{outdir}/curves_x.npy", np.array(curves_x, dtype=object))
        np.save(f"{outdir}/curves_p.npy", np.array(curves_p, dtype=object))
        np.save(f"{outdir}/crit_ps.npy", np.array(crit_ps, dtype=object))
        np.save(f"{outdir}/eigs.npy", np.array(eigs, dtype=object))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

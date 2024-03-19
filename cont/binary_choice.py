"""Bifurcation diagram for the binary choice landscape (phi1).

"""

import sys
import argparse
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

H = lambda x: -np.array([
    [[24*x[0], -8.],
     [-8., 0.]],
    [[-8., 0.],
     [0., 24*x[1]+6.]]
])


XSTARTS = [
    # [[ 0.,  0.01], 'b'],
    [[ 0.1891, -0.1581], 'b'],
    # [[-0.1891, -0.1581], 'b'],
    [[ 0.9201, 0.6418], 'brown'],
    [[-0.9201, 0.6418], 'g'],
]

P1 = lambda x: 4*x[0]**3 - 8*x[0]*x[1]
P2 = lambda x: -4*x[1]**3 - 3*x[1]**2 + 4*x[0]**2 - 2*x[1]

maxiter = 100000
ds = 1e-3
min_ds = 1e-8
max_ds = 1e-1
max_delta_p = 1e-1
rho = 1e-1
P1LIMS = [-20, 20]
P2LIMS = [-20, 20]
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


def plot_binary_choice_bifurcation_diagram(
        xstarts=XSTARTS,
        plot_starts=False,
        plot_first_steps=False,
        plot_failed_to_converge_points=False,
        ax=None,
        figsize=(4,4),
        saveas="",
        show=False,
        tight_layout=True,
        xlabel='$p_1$',
        ylabel='$p_2$',
        p1_view_lims=P1_VIEW_LIMS,
        p2_view_lims=P2_VIEW_LIMS,
        verbosity=0,
):
    """Plot a bifurcation diagram for the binary choice landscape.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    curves_x = []
    curves_p = []
    crit_ps = []
    eigs = []
    failed_to_converge_xs = []
    failed_to_converge_ps = []
    for i in range(len(xstarts)):
        x0 = np.array(xstarts[i][0])
        col = xstarts[i][1]
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
                verbosity=verbosity,

            )
            curves_x.append(xs)
            curves_p.append(ps)
            crit_ps.append(cps)
            eigs.append(np.array(d['eigs']))
            failed_to_converge_ps.append(np.array(d['failed_to_converge_ps']))
            failed_to_converge_xs.append(np.array(d['failed_to_converge_xs']))

            if plot_starts:
                ax.plot(ps[0,0], ps[0,1], 'o', alpha=0.2, color=col)
            
            if len(ps) > 1:
                ax.plot(ps[:,0], ps[:,1], '-', alpha=1, color=col)
                if plot_first_steps:
                    ax.plot(ps[1,0], ps[1,1], '*', alpha=0.6, color=col)

            if plot_failed_to_converge_points:
                for p in d['failed_to_converge_ps']:
                    ax.plot(*p, '^', alpha=0.6, color=col)

    ax.set_xlim(*p1_view_lims)
    ax.set_ylim(*p2_view_lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if tight_layout: plt.tight_layout()
    if saveas: plt.savefig(saveas, bbox_inches='tight')
    if show: plt.show()
    return ax


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_estimates', action="store_true")
    parser.add_argument('--plot_starts', action="store_true")
    parser.add_argument('--plot_first_steps', action="store_true")
    parser.add_argument('--plot_failed_to_converge_points', action="store_true")
    parser.add_argument('--plot_critical_ps', action="store_true")
    parser.add_argument('--plot_vecs', action="store_true")
    parser.add_argument('--show', action="store_true")
    parser.add_argument('-s', '--saveas', type=str, 
                        default="bifcurves_binary_choice.pdf")
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    return parser.parse_args(args)


def main(args):
    plot_estimates = args.plot_estimates
    plot_starts = args.plot_starts
    plot_first_steps = args.plot_first_steps
    plot_failed_to_converge_points = args.plot_failed_to_converge_points
    plot_critical_ps = args.plot_critical_ps
    plot_vecs = args.plot_vecs
    verbosity = args.verbosity

    fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    failed_to_converge_xs = []
    failed_to_converge_ps = []
    failed_to_converge_flags = []
    failed_to_converge_reasons = []
    for i in range(len(XSTARTS)):
        x0 = np.array(XSTARTS[i][0])
        col = XSTARTS[i][1]
        p0 = np.array([P1(x0), P2(x0)])
        xs, ps, d = trace_curve(
            x0, p0, F, Fx, dxFxPhi, Fp,
            maxiter=maxiter, 
            ds=ds,
            min_ds=min_ds,
            max_ds=max_ds,
            max_delta_p=max_delta_p,
            rho=rho,
            plims=[P1LIMS, P2LIMS],
            verbosity=verbosity,
        )
        failed_to_converge_ps.append(np.array(d['failed_to_converge_ps']))
        failed_to_converge_xs.append(np.array(d['failed_to_converge_xs']))
        failed_to_converge_flags.append(np.array(d['failed_to_converge_flags']))
        failed_to_converge_reasons.append(np.array(d['failed_to_converge_reasons']))

        p0_est = d['p0_est']
        x0_est = d['x0_est']
        if plot_estimates:
            ax1.plot(p0_est[0], p0_est[1], 'o', alpha=0.2, color=col)
            ax2.plot(x0_est[0], x0_est[1], 'o', alpha=0.2, color=col)
        
        p0 = d['p0']
        x0 = d['x0']
        if plot_starts:
            ax1.plot(p0[0], p0[1], 'o', alpha=0.9, color=col)
            ax2.plot(x0[0], x0[1], 'o', alpha=0.9, color=col)

        ps_fwd = d['ps_path_fwd']
        xs_fwd = d['xs_path_fwd']
        ps_rev = d['ps_path_rev']
        xs_rev = d['xs_path_rev']
        if plot_first_steps:
            if len(ps_fwd) > 0:
                p1_fwd = ps_fwd[1]
                x1_fwd = xs_fwd[1]
                ax1.plot(p1_fwd[0], p1_fwd[1], '*', alpha=0.6, color=col)
                ax2.plot(x1_fwd[0], x1_fwd[1], '*', alpha=0.6, color=col)
            if len(ps_rev) > 0:
                p1_rev = ps_rev[1]
                x1_rev = xs_rev[1]
                ax1.plot(p1_rev[0], p1_rev[1], '*', alpha=0.6, color=col)
                ax2.plot(x1_rev[0], x1_rev[1], '*', alpha=0.6, color=col)

        if len(ps_fwd) > 1:
            ax1.plot(ps_fwd[:,0], ps_fwd[:,1], '-', alpha=1.0, color=col)
            ax2.plot(xs_fwd[:,0], xs_fwd[:,1], '-', alpha=1.0, color=col)
        
        if len(ps_rev) > 1:
            ax1.plot(ps_rev[:,0], ps_rev[:,1], '-', alpha=1.0, color=col)
            ax2.plot(xs_rev[:,0], xs_rev[:,1], '-', alpha=1.0, color=col)

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


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

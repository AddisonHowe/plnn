"""Bifurcation diagram for the binary flip landscape (phi2).

"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cont.continuation import trace_curve

F = lambda x, p: -np.array([
        4*x[0]**3 + 3*x[0]**2 - 2*x[1]**2 - 2*x[0] + p[0], 
        4*x[1]**3 - 4*x[0]*x[1] + p[1],
    ])

Fx = lambda x: np.array([
        [-12*x[0]**2 - 6*x[0] + 2,      4*x[1]],
        [4*x[1],                        4*x[0] - 12*x[1]**2],
    ])

dxFxPhi = lambda x, phi: np.array([
        [-(24*x[0]+6)*phi[0],   4*phi[1]],
        [4*phi[1],              4*phi[0]-24*x[1]*phi[1]],
    ])

Fp = lambda x, p: np.array([[-1, 0], [0, -1]])

XSTARTS = [
    [[-0.811,  0.965], 'brown'],
    [[-0.811, -0.965], 'brown'],
    [[0.147,  0.139], 'k'],
    [[0.147, -0.139], 'k'],
    [[ 0.050, -0.096], 'k',],    
    [[0.896,  0.577], 'cyan'],
    [[0.345,  1.034], 'cyan'],
    [[0.896, -0.577], 'pink'],
    [[0.345, -1.034], 'pink'],
]

P1 = lambda x: -4*x[0]**3 - 3*x[0]**2 + 2*x[1]**2 + 2*x[0]
P2 = lambda x: 4*x[0]*x[1] - 4*x[1]**3

maxiter = 10000
ds = 1e-3
min_ds = 1e-8
max_ds = 1e-1
max_delta_p = 1e-2
rho = 1e-1
P1LIMS = [-10, 10]
P2LIMS = [-10, 10]
P1_VIEW_LIMS = [-2, 1.5]
P2_VIEW_LIMS = [-1.5, 1.5]

def get_binary_flip_curves(p1lims=P1LIMS, p2lims=P2LIMS, xstarts=XSTARTS):
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


def plot_binary_flip_bifurcation_diagram(
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
    """Plot a bifurcation diagram for the binary flip landscape.
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
    parser.add_argument('--plot_starts', action="store_true")
    parser.add_argument('--plot_first_steps', action="store_true")
    parser.add_argument('--plot_failed_to_converge_points', action="store_true")
    parser.add_argument('--show', action="store_true")
    parser.add_argument('-s', '--saveas', type=str, 
                        default="bifcurves_binary_flip.pdf")
    return parser.parse_args(args)


def main(args):
    plot_starts = args.plot_starts
    plot_first_steps = args.plot_first_steps
    plot_failed_to_converge_points =args.plot_failed_to_converge_points

    fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    curves_x = []
    curves_p = []
    crit_ps = []
    eigs = []
    failed_to_converge_xs = []
    failed_to_converge_ps = []
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


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

"""

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from .continuation import trace_curve


def get_fold_curves(
        F, Fx, dxFxPhi, Fp,
        xstarts, 
        p1_func, 
        p2_func,
        p1lims=[-10, 10], 
        p2lims=[-10, 10], 
        maxiter=10000,
        ds=1e-4, 
        min_ds=1e-8, 
        max_ds=1e-2,
        max_delta_p=1e-2,
        rho=1e-1,
        verbosity=0,
        rng=None,
        seed=None,
):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    curves_p = []
    colors = []
    for i in range(len(xstarts)):
        x0 = np.array(xstarts[i][0])
        col = xstarts[i][1]
        p0 = np.array([p1_func(x0), p2_func(x0)])
        _, ps, _ = trace_curve(
            x0, p0, F, Fx, dxFxPhi, Fp,
            maxiter=maxiter, 
            ds=ds,
            min_ds=min_ds,
            max_ds=max_ds,
            max_delta_p=max_delta_p,
            rho=rho,
            plims=[p1lims, p2lims],
            verbosity=verbosity,
            rng=rng,
        )
        if len(ps) > 0:
            curves_p.append(ps)
            colors.append(col)
    return curves_p, colors


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_estimates', action="store_true")
    parser.add_argument('--plot_starts', action="store_true")
    parser.add_argument('--plot_first_steps', action="store_true")
    parser.add_argument('--plot_failed_to_converge_points', action="store_true")
    parser.add_argument('--plot_critical_ps', action="store_true")
    parser.add_argument('--plot_vecs', action="store_true")
    parser.add_argument('--show', action="store_true")
    parser.add_argument('--outdir', type=str, default="")
    parser.add_argument('-s', '--saveas', type=str, default="")
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    return parser


def plot_diagrams(
        argdict, 
        xstarts, p1func, p2func, 
        F, J, dxFxPhi, Fp,
        maxiter, ds, min_ds, max_ds, max_delta_p, rho, p1lims, p2lims,
        p1_view_lims, p2_view_lims, x_view_lims, y_view_lims,
):
    plot_estimates = argdict['plot_estimates']
    plot_starts = argdict['plot_starts']
    plot_first_steps = argdict['plot_first_steps']
    plot_failed_to_converge_points = argdict['plot_failed_to_converge_points']
    plot_critical_ps = argdict['plot_critical_ps']
    plot_vecs = argdict['plot_vecs']
    verbosity = argdict['verbosity']
    saveas = argdict['saveas']
    show = argdict['show']

    fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    failed_to_converge_xs = []
    failed_to_converge_ps = []
    failed_to_converge_flags = []
    failed_to_converge_reasons = []

    for i in range(len(xstarts)):
        x0 = np.array(xstarts[i][0])
        col = xstarts[i][1]
        p0 = np.array([p1func(x0), p2func(x0)])
        xs, ps, d = trace_curve(
            x0, p0, F, J, dxFxPhi, Fp,
            maxiter=maxiter, 
            ds=ds,
            min_ds=min_ds,
            max_ds=max_ds,
            max_delta_p=max_delta_p,
            rho=rho,
            plims=[p1lims, p2lims],
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
            
        # if plot_vecs:
        #     vnormx = 1e-2
        #     tauxs = np.array(d['tauxs'])
        #     vx0 = xs
        #     vx1 = xs + vnormx * tauxs
        #     vnormp = 1e-2
        #     taups = np.array(d['taups'])
        #     vp0 = ps
        #     vp1 = ps + vnormp * taups
        #     ax1.plot(
        #         [vp0[:,0], vp1[:,0]], [vp0[:,1], vp1[:,1]], 
        #         '-', alpha=0.5, color='r'
        #     )
        #     ax2.plot(
        #         [vx0[:,0], vx1[:,0]], [vx0[:,1], vx1[:,1]], 
        #         '-', alpha=0.5, color='r'
        #     )
            
        # if plot_failed_to_converge_points:
        #     for j, p in enumerate(d['failed_to_converge_ps']):
        #         ax1.plot(*p, '^', alpha=0.6, color=col)
        #         ax1.annotate(d['failed_to_converge_flags'][j], p)
        #     for j, x in enumerate(d['failed_to_converge_xs']):
        #         ax2.plot(*x, '^', alpha=0.6, color=col)
        #         ax2.annotate(d['failed_to_converge_flags'][j], x)

        # if plot_critical_ps:
        #     for j, p in enumerate(d['critical_ps']):
        #         ax1.plot(*p, '+', alpha=0.6, color=col)
        #         ax1.annotate("c", p)
        #     for j, x in enumerate(d['critical_xs']):
        #         ax2.plot(*x, '+', alpha=0.6, color=col)
        #         ax2.annotate("c", x)

    ax1.set_xlim(*p1_view_lims)
    ax1.set_ylim(*p2_view_lims)
    ax1.set_xlabel('$p_1$')
    ax1.set_ylabel('$p_2$')

    ax2.set_xlim(*x_view_lims)
    ax2.set_ylim(*y_view_lims)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    plt.tight_layout()

    if saveas:
        plt.savefig(saveas, bbox_inches='tight')

    if show:
        plt.show()

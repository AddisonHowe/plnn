"""PLNN plotting methods

"""

import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.spatial import Delaunay
import jax.numpy as jnp

from .config import DEFAULT_CMAP, SIG1_COLOR, SIG2_COLOR


def plot_phi(
        model,
        tilt=None,
        signal=None,
        sigparams=None,
        eval_time=None,
        r=4, 
        xrange=None,
        yrange=None,
        res=50, 
        plot3d=False, 
        lognormalize=True, 
        normalize=False,
        minimum=None,
        clip=None,
        include_tilt_inset=False,
        inset_scale='30%',
        inset_loc='lower left',
        inset_bbox_to_anchor=None,
        tilt_arrow_width=0.001,
        signal1_color=SIG1_COLOR,
        signal2_color=SIG2_COLOR,
        xlims=None, 
        ylims=None, 
        zlims=None, 
        xlabel="$x$", 
        ylabel="$y$", 
        zlabel="$z$",
        xticks=None,
        yticks=None,
        zticks=None,
        title="$\\phi(x,y)$",
        title_fontsize=None,
        include_cbar=True,
        cbar_title="$\phi$", 
        cbar_titlefontsize=6,
        cbar_ticklabelsize=6,
        cmap=DEFAULT_CMAP, 
        ncontours=0,
        contour_linewidth=1,
        contour_linestyle='solid',
        contour_linecolor='k',
        contour_linealpha=1.0,
        figsize=(6, 4),
        view_init=(30, -45),
        alpha=1.0,
        ax=None,
        equal_axes=True,
        tight_layout=True,
        saveas=None,
        show=False,
        hull=None,
):
    """Plot the scalar function phi defined by a given model.

    Returns:
        Axis object.
    """
    if ax is None and plot3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
    elif ax is None and (not plot3d):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if equal_axes:
        ax.set_aspect('equal')

    if xrange is None:
        xrange = [-r, r]
    if yrange is None:
        yrange = [-r, r]

    # Get grid
    x = np.linspace(*xrange, res)
    y = np.linspace(*yrange, res)
    xs, ys = np.meshgrid(x, y)
    z = np.array([xs.flatten(), ys.flatten()]).T
    z = jnp.array(z, dtype=jnp.float64)

    # Determine tilt based on given information
    if tilt is not None:
        # Tilt given directly
        phi = model.phi_with_tilts(eval_time, z, jnp.array(tilt))
    elif signal is not None:
        # Signal values given directly
        phi = model.phi_with_signal(eval_time, z, jnp.array(signal))
    elif (sigparams is not None) and (eval_time is not None):
        # Signal parameters and evaluation time given.
        phi = model.tilted_phi(eval_time, z, jnp.array(sigparams))
    else:
        # Plot landscape with no tilt
        if model.signal_type == 'jump':
            sigparams = model.nsigs * [[1, 0, 0]]
        elif model.signal_type == 'sigmoid':
            sigparams = model.nsigs * [[1, 0, 0, 0]]
        phi = model.phi(z)
    
    # Convert phi to a numpy array
    phi = np.array(phi)
    
    # Hull Restriction
    within_hull = np.ones(z.shape[0], dtype=bool)
    if hull is not None:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        within_hull = hull.find_simplex(z)>=0  # screen for xy inside given hull
    phi_within_hull = phi[within_hull]

    # Normalization
    if lognormalize:
        phi = np.log(1 + phi - phi_within_hull.min())
        phi_within_hull = np.log(1 + phi_within_hull - phi_within_hull.min())
    elif normalize:
        phi = 1 + phi - phi_within_hull.min()  # set minimum to 1
        phi_within_hull = 1 + phi_within_hull - phi_within_hull.min()
    if minimum is not None:
        phi = phi - (phi_within_hull.min() - minimum)  # set minimum to given value
        phi_within_hull = phi_within_hull - (phi_within_hull.min() - minimum)

    # Clipping
    clip = 1 + phi_within_hull.max() if clip is None else clip
    if clip < phi_within_hull.min():
        warnings.warn(f"Clip value {clip} is below minimum value to plot.")
        clip = 1 + phi_within_hull.max()
    under_cutoff = phi <= clip

    plot_screen = np.ones(under_cutoff.shape)
    plot_screen[~under_cutoff] = np.nan
    plot_screen[~within_hull] = np.nan
    phi_plot = phi * plot_screen
    phi_plot = phi_plot.reshape(xs.shape)

    # Get levelsets
    if isinstance(ncontours, int) and ncontours > 0:
        plot_contours = True
        yidx = int(len(phi_plot[0]) // 2)
        idxs = np.linspace(
            0, len(phi_plot[0]), 
            ncontours, 
            endpoint=False,
            dtype=int, 
        )
        levels = np.sort(phi_plot[yidx, idxs])
    elif isinstance(ncontours, list):
        plot_contours = True
        raise NotImplementedError()
    else:
        plot_contours = False

    # Plot landscape or heatmap of phi
    if plot3d:
        sc = ax.plot_surface(
            xs, ys, phi_plot, 
            vmin=phi[under_cutoff & within_hull].min(),
            vmax=phi[under_cutoff & within_hull].max(),
            cmap=cmap,
            alpha=alpha,
        )
        if plot_contours:
            ax.contour(
                xs, ys, phi_plot,
                levels=levels, 
                vmin=phi[under_cutoff & within_hull].min(),
                vmax=phi[under_cutoff & within_hull].max(),
                cmap=cmap,
                linewidths=contour_linewidth,
                linestyles=contour_linestyle, 
                offset=0,
            )
    else:
        sc = ax.pcolormesh(
            xs, ys, phi_plot,
            vmin=phi[under_cutoff & within_hull].min(),
            vmax=phi[under_cutoff & within_hull].max(),
            cmap=cmap, 
            shading="gouraud",
        )
        if plot_contours:
            ax.contour(
                xs, ys, phi_plot,
                levels=levels, 
                vmin=phi[under_cutoff & within_hull].min(),
                vmax=phi[under_cutoff & within_hull].max(),
                alpha=contour_linealpha,
                colors=contour_linecolor,
                linewidths=contour_linewidth,
                linestyles=contour_linestyle, 
            )
    
    # Plot signal effect inset
    if include_tilt_inset and not plot3d:
        tilt_weights = model.get_parameters()['tilt.w'][0]
        subax = inset_axes(ax,
            width=inset_scale,
            height=inset_scale,
            loc=inset_loc,
            bbox_to_anchor=inset_bbox_to_anchor,
            bbox_transform=ax.transAxes,
        )
        subax.set_aspect('equal')
        subax.axis('off')
        subax.set_xlim([-1.25, 1.25])
        subax.set_ylim([-1.25, 1.25])
        scale = np.max(np.linalg.norm(tilt_weights, axis=0))
        subax.arrow(
            0, 0, -tilt_weights[0,0]/scale, -tilt_weights[1,0]/scale, 
            width=tilt_arrow_width, 
            head_width=100*tilt_arrow_width, 
            length_includes_head=True, 
            fc=signal1_color, 
            ec=signal1_color,
            label="$s_1$"
        )
        subax.arrow(
            0, 0, -tilt_weights[0,1]/scale, -tilt_weights[1,1]/scale, 
            width=tilt_arrow_width, 
            head_width=100*tilt_arrow_width, 
            length_includes_head=True, 
            fc=signal2_color, 
            ec=signal2_color,
            label="$s_2$"
        )
        

    # Colorbar
    fig = ax.figure
    if include_cbar:
        if plot3d:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
            cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
            cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
    
    # Format plot
    if xlims is not None: ax.set_xlim(*xlims)
    if ylims is not None: ax.set_ylim(*ylims)
    if zlims is not None: ax.set_zlim(*zlims)
    if title_fontsize is None:
        title_fontsize = plt.rcParams['axes.titlesize']
    if title is not None: ax.set_title(title, size=title_fontsize)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if xticks is False: ax.set_xticks([])
    if yticks is False: ax.set_yticks([])
    if zticks is False: ax.set_zticks([])
    if plot3d: 
        ax.set_zlabel(zlabel)
        ax.view_init(*view_init)
    
    # Save and close
    if tight_layout: plt.tight_layout()
    if saveas: plt.savefig(saveas, bbox_inches='tight')
    if not show: plt.close()
    return ax


def plot_f(
        model, 
        tilt=None,
        signal=None,
        sigparams=None,
        eval_time=None,
        r=4, 
        xrange=None,
        yrange=None,
        res=50, 
        xlims=None, 
        ylims=None, 
        xlabel="$x$", 
        ylabel="$y$", 
        xticks=None,
        yticks=None,
        title="$\\phi(x,y)$",
        title_fontsize=None,
        include_cbar=True,
        cbar_title="$\phi$", 
        cbar_titlefontsize=6,
        cbar_ticklabelsize=6,
        cmap=DEFAULT_CMAP, 
        figsize=(6, 4),
        ax=None,
        equal_axes=True,
        tight_layout=True,
        saveas=None,
        show=False,
):
    """Plot the vector field f defined by the given model.

    Returns:
        Axis object.
    """
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if equal_axes:
        ax.set_aspect('equal')

    if xrange is None:
        xrange = [-r, r]
    if yrange is None:
        yrange = [-r, r]

    # Get grid
    x = np.linspace(*xrange, res)
    y = np.linspace(*yrange, res)
    xs, ys = np.meshgrid(x, y)
    z = np.array([xs.flatten(), ys.flatten()]).T
    z = jnp.array(z, dtype=jnp.float64)
    
    # Determine tilt based on given information
    if tilt is not None:
        # Tilt given directly
        grad_phi = model.grad_phi_with_tilts(eval_time, z, jnp.array(tilt))
    elif signal is not None:
        # Signal values given directly
        grad_phi = model.grad_phi_with_signal(eval_time, z, jnp.array(signal))
    elif (sigparams is not None) and (eval_time is not None):
        # Signal parameters and evaluation time given.
        grad_phi = model.tilted_grad_phi(eval_time, z, jnp.array(sigparams))
    else:
        # Plot vector field with no tilt
        if model.signal_type == 'jump':
            sigparams = model.nsigs * [[1, 0, 0]]
        elif model.signal_type == 'sigmoid':
            sigparams = model.nsigs * [[1, 0, 0, 0]]
        grad_phi = model.grad_phi(eval_time, z)

    # Compute f as the negative of the gradient
    f = -np.array(grad_phi)
    fu, fv = f.T
    fnorms = np.sqrt(fu**2 + fv**2)

    # Plot force field, tilted by signals
    sc = ax.quiver(
        xs, ys, 
        fu / fnorms, fv / fnorms, 
        fnorms, 
        cmap=cmap
    )
    
    # Colorbar
    if include_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig = ax.figure
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
        cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
    
    # Format plot
    if xlims is not None: ax.set_xlim(*xlims)
    if ylims is not None: ax.set_ylim(*ylims)
    if title_fontsize is None:
        title_fontsize = plt.rcParams['axes.titlesize']
    if title is not None: ax.set_title(title, size=title_fontsize)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if xticks is False: ax.set_xticks([])
    if yticks is False: ax.set_yticks([])
    
    # Save and close
    if tight_layout: plt.tight_layout()
    if saveas: plt.savefig(saveas, bbox_inches='tight')
    if not show: plt.close()
    return ax

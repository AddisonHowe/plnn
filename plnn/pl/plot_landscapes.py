"""Plotting functions for in silico landscapes.

"""

import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from .config import DEFAULT_CMAP

def func_phi1(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y - p1*x + p2*y

def func_phi2(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

def plot_landscape(
        phi_func, 
        params=[0,0], 
        r=2, 
        res=100, 
        plot3d=False,
        lognormalize=True, 
        normalize=False,
        minimum=None,
        clip=None,
        xlims=None, 
        ylims=None, 
        zlims=None, 
        xlabel="$x$", 
        ylabel="$y$", 
        zlabel="$z$",
        xticks=None,
        yticks=None,
        zticks=None,
        title="$\phi$",
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
    ):
    """Plot the landscape phi.
    
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

    # Get grid
    x = np.linspace(-r, r, res)
    y = np.linspace(-r, r, res)
    xs, ys = np.meshgrid(x, y)

    # Compute phi over the grid
    phi = phi_func(xs, ys, *params)

    # Normalization
    if lognormalize:
        phi = np.log(1 + phi - phi.min())
    elif normalize:
        phi = 1 + phi - phi.min()  # set minimum to 1
    if minimum is not None:
        phi = phi - (phi.min() - minimum)  # set minimum to given value

    # Clipping
    clip = 1 + phi.max() if clip is None else clip
    if clip < phi.min():
        warnings.warn(f"Clip value {clip} is below minimum value to plot.")
        clip = phi.max()
    under_cutoff = phi <= clip
    plot_screen = np.ones(under_cutoff.shape)
    plot_screen[~under_cutoff] = np.nan
    phi_plot = phi * plot_screen

    # Get levelsets
    if ncontours:
        yidx = int(len(phi_plot[0]) // 2)
        idxs = np.linspace(
            0, len(phi_plot[0]), 
            ncontours, 
            endpoint=False,
            dtype=int, 
        )
        levels = np.sort(phi_plot[yidx, idxs])

    # Plot landscape or heatmap of phi
    if plot3d:
        sc = ax.plot_surface(
            xs, ys, phi_plot.reshape(xs.shape),
            vmin=phi[under_cutoff].min(),
            vmax=phi[under_cutoff].max(),
            cmap=cmap,
            alpha=alpha,
        )
        if ncontours:
            ax.contour(
                xs, ys, phi_plot.reshape(xs.shape),
                levels=levels, 
                vmin=phi[under_cutoff].min(),
                vmax=phi[under_cutoff].max(),
                cmap=cmap,
                linewidths=contour_linewidth,
                linestyles=contour_linestyle, 
                offset=0,
            )
    else:
        sc = ax.pcolormesh(
            xs, ys, phi_plot.reshape(xs.shape),
            vmin=phi[under_cutoff].min(),
            vmax=phi[under_cutoff].max(),
            cmap=cmap, 
            shading="gouraud",
        )
        if ncontours:
            ax.contour(
                xs, ys, phi_plot.reshape(xs.shape),
                levels=levels, 
                vmin=phi[under_cutoff].min(),
                vmax=phi[under_cutoff].max(),
                alpha=contour_linealpha,
                colors=contour_linecolor,
                linewidths=contour_linewidth,
                linestyles=contour_linestyle, 
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
    if title: ax.set_title(title, size=title_fontsize)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
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

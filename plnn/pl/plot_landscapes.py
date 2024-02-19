"""Plotting functions for in silico landscapes.

"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch 

def func_phi1(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y - p1*x + p2*y

def func_phi2(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

def plot_landscape(
        phi_func, 
        r=2, 
        res=100, 
        plot3d=False,
        params=[0,0], 
        lognormalize=True, 
        cmap='coolwarm', 
        xlims=None, 
        ylims=None, 
        xlabel="$x$", 
        ylabel="$y$", 
        zlabel="$z$",
        title="$\phi$",
        cbar_title="$\phi$", 
        ax=None,
        figsize=(6, 4),
        view_init=(30, -45),
        clip=None,
        equal_axes=True,
        saveas=None,
    ):

    if ax is None and plot3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    elif ax is None and (not plot3d):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if equal_axes:
        ax.set_aspect('equal')

    x = np.linspace(-r, r, res)
    y = np.linspace(-r, r, res)
    xs, ys = np.meshgrid(x, y)
    z = np.array([xs.flatten(), ys.flatten()]).T
    z = torch.tensor(z, dtype=torch.float32, requires_grad=True)[None,:]

    phi_star = phi_func(xs, ys, params[0], params[1])
    if lognormalize:
        phi_star = np.log(1 + phi_star - phi_star.min())  # log normalize

    clip = phi_star.max() + 1 if clip is None else clip
    under_cutoff = phi_star <= clip
    plot_screen = np.ones(under_cutoff.shape)
    plot_screen[~under_cutoff] = np.nan
    phi_star_plot = phi_star * plot_screen

    # Plot
    if plot3d:
        sc = ax.plot_surface(
            xs, ys, phi_star_plot.reshape(xs.shape),
            vmin=phi_star[under_cutoff].min(),
            vmax=phi_star[under_cutoff].max(),
            cmap=cmap
        )
    else:
        sc = ax.pcolormesh(
            xs, ys, phi_star_plot.reshape(xs.shape),
            vmin=phi_star[under_cutoff].min(),
            vmax=phi_star[under_cutoff].max(),
            cmap=cmap, 
        )

    fig = ax.figure
    if plot3d:
        # divider = make_axes_locatable(ax)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.set_title(cbar_title, size=8)
        cbar.ax.tick_params(labelsize=8)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_title(cbar_title, size=8)
        cbar.ax.tick_params(labelsize=8)


    # Format plot
    if xlims is not None: ax.set_xlim(*xlims)
    if ylims is not None: ax.set_ylim(*ylims)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot3d: 
        ax.set_zlabel(zlabel)
        ax.view_init(*view_init)
    plt.tight_layout()
    
    if saveas: plt.savefig(saveas)
    
    return ax

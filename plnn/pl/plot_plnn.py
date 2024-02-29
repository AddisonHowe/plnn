"""PLNN plotting methods

"""

import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import jax.numpy as jnp

from .config import DEFAULT_CMAP


def plot_phi(
        model, 
        tilt=None,
        signal=None,
        sigparams=None,
        eval_time=None,
        r=4, 
        res=50, 
        plot3d=False, 
        **kwargs
):
    """Plot the scalar function phi.

    Args:
        r (int) : 
        res (int) :
        plot3d (bool) :
        normalize (bool) :
        log_normalize (bool) :
        clip (float) :
        ax (Axis) :
        figsize (tuple[float]) :
        xlims (tuple[float]) :
        ylims (tuple[float]) :
        xlabel (str) :
        ylabel (str) :
        zlabel (str) :
        title (str) :
        cmap (Colormap) :
        include_cbar (bool) :
        cbar_title (str) :
        cbar_titlefontsize (int) :
        cbar_ticklabelsize (int) :
        view_init (tuple) :
        saveas (str) :
        show (bool) :

    Returns:
        Axis object.
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    normalize = kwargs.get('normalize', True)
    log_normalize = kwargs.get('log_normalize', True)
    clip = kwargs.get('clip', None)
    ax = kwargs.get('ax', None)
    figsize = kwargs.get('figsize', (6, 4))
    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)
    xlabel = kwargs.get('xlabel', "$x$")
    ylabel = kwargs.get('ylabel', "$y$")
    zlabel = kwargs.get('zlabel', "$\\phi$")
    title = kwargs.get('title', "$\\phi(x,y)$")
    cmap = kwargs.get('cmap', DEFAULT_CMAP)
    include_cbar = kwargs.get('include_cbar', True)
    cbar_title = kwargs.get('cbar_title', 
                            "$\\ln\\phi$" if log_normalize else "$\\phi$")
    cbar_titlefontsize = kwargs.get('cbar_titlefontsize', 10)
    cbar_ticklabelsize = kwargs.get('cbar_ticklabelsize', 8)
    view_init = kwargs.get('view_init', (30, -45))
    tight_layout = kwargs.get('tight_layout', True)
    equal_axes = kwargs.get('equal_axes', False)
    saveas = kwargs.get('saveas', None)
    show = kwargs.get('show', False)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if ax is None and plot3d:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    elif ax is None and (not plot3d):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if equal_axes:
        ax.set_aspect('equal')

    # Get grid
    x = np.linspace(-r, r, res)
    y = np.linspace(-r, r, res)
    xs, ys = np.meshgrid(x, y)
    z = np.array([xs.flatten(), ys.flatten()]).T
    z = jnp.array(z, dtype=jnp.float32)

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

    # Convert phi to an array
    phi = np.array(phi)
    
    # Normalization
    if normalize:
        phi = 1 + phi - phi.min()  # set minimum to 1
    if log_normalize:
        phi = np.log(phi)

    # Clipping
    clip = 1 + phi.max() if clip is None else clip
    if clip < phi.min():
        warnings.warn(f"Clip value {clip} is below minimum value to plot.")
        clip = phi.max()
    under_cutoff = phi <= clip
    plot_screen = np.ones(under_cutoff.shape)
    plot_screen[~under_cutoff] = np.nan
    phi_plot = phi * plot_screen

    # Plot phi
    if plot3d:
        sc = ax.plot_surface(
            xs, ys, phi_plot.reshape(xs.shape), 
            vmin=phi[under_cutoff].min(),
            vmax=phi[under_cutoff].max(),
            cmap=cmap
        )
    else:
        sc = ax.pcolormesh(
            xs, ys, phi_plot.reshape(xs.shape),
            vmin=phi[under_cutoff].min(),
            vmax=phi[under_cutoff].max(),
            cmap=cmap, 
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
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if plot3d: 
        ax.set_zlabel(zlabel)
        ax.view_init(*view_init)
    
    if tight_layout: plt.tight_layout()
    
    # Save and close
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
        res=50, 
        **kwargs):
    """Plot the vector field f.
    Args:
        signal (float or tuple[float]) :
        r (int) : 
        res (int) :
        ax (Axis) :
        figsize (tuple[float]) :
        xlims (tuple[float]) :
        ylims (tuple[float]) :
        xlabel (str) :
        ylabel (str) :
        title (str) :
        cmap (Colormap) :
        include_cbar (bool) :
        cbar_title (str) :
        cbar_titlefontsize (int) :
        cbar_ticklabelsize (int) :
        saveas (str) :
        show (bool) :
    Returns:
        Axis object.
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    ax = kwargs.get('ax', None)
    figsize = kwargs.get('figsize', (6, 4))
    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)
    xlabel = kwargs.get('xlabel', "$x$")
    ylabel = kwargs.get('ylabel', "$y$")
    title = kwargs.get('title', "$f(x,y|\\vec{s})$")
    equal_axes = kwargs.get('equal_axes', False)
    cmap = kwargs.get('cmap', DEFAULT_CMAP)
    include_cbar = kwargs.get('include_cbar', True)
    cbar_title = kwargs.get('cbar_title', "$|f|$")
    cbar_titlefontsize = kwargs.get('cbar_titlefontsize', 10)
    cbar_ticklabelsize = kwargs.get('cbar_ticklabelsize', 8)
    saveas = kwargs.get('saveas', None)
    show = kwargs.get('show', False)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)

    if equal_axes:
        ax.set_aspect('equal')

    # Get grid
    x = np.linspace(-r, r, res)
    y = np.linspace(-r, r, res)
    xs, ys = np.meshgrid(x, y)
    z = np.array([xs.flatten(), ys.flatten()]).T
    z = jnp.array(z, dtype=jnp.float32)
    
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
    sc = ax.quiver(xs, ys, fu/fnorms, fv/fnorms, fnorms, cmap=cmap)
    
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
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
    # Save and close
    if saveas: plt.savefig(saveas, bbox_inches='tight')
    if not show: plt.close()
    return ax
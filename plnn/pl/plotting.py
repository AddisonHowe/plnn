"""Plotting functions

"""

import numpy as np
import matplotlib.pyplot as plt

from .plot_landscapes import func_phi1, func_phi2, plot_landscape


def plot_training_loss_history(
        loss_hist_train, startidx=0, 
        log=False, title="Training Loss", 
        saveas=None, ax=None, 
        **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize'))
    erange = np.arange(startidx, len(loss_hist_train))
    fplot = ax.semilogy if log else ax.plot
    fplot(
        1 + erange, loss_hist_train[erange], 
        linestyle=kwargs.get('linestyle'),
        marker=kwargs.get('marker'),
        color=kwargs.get('color')
    )
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax


def plot_validation_loss_history(
        loss_hist_valid, startidx=0, 
        log=False, title="Validation Loss", 
        saveas=None, ax=None, optidx=None,
        **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize'))
    erange = np.arange(startidx, len(loss_hist_valid))
    fplot = ax.semilogy if log else ax.plot
    fplot(
        1 + erange, loss_hist_valid[erange], 
        linestyle=kwargs.get('linestyle'),
        marker=kwargs.get('marker'),
        color=kwargs.get('color')
    )
    if optidx:
        ax.axvline(optidx, color='k', linestyle='--', alpha=0.7)
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax


def plot_loss_history(
        loss_hist_train, 
        loss_hist_valid, 
        startidx=0, 
        log=False, 
        title="Loss History", 
        color_train='r',
        color_valid='b',
        marker_train='.',
        marker_valid='.',
        linestyle_train='-',
        linestyle_valid='-',
        linewidth_train=1,
        linewidth_valid=1,
        alpha_train=0.9,
        alpha_valid=0.5,
        saveas=None,
        ax=None,
        **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize'))
    erange = np.arange(startidx, len(loss_hist_train))
    fplot = ax.semilogy if log else ax.plot
    fplot(
        1 + erange, loss_hist_train[erange], 
        color=color_train,
        marker=marker_train,
        linestyle=linestyle_train,
        linewidth=linewidth_train,
        label="Training", alpha=alpha_train
    )
    fplot(
        1 + erange, loss_hist_valid[erange], 
        color=color_valid,
        marker=marker_valid,
        linestyle=linestyle_valid,
        linewidth=linewidth_valid,
        label="Validation", alpha=alpha_valid
    )
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(title)
    ax.legend()
    if saveas: plt.savefig(saveas)
    return ax


def plot_train_vs_valid_history(
        loss_hist_train, loss_hist_valid, startidx=0,
        log=True, title="Training vs Validation Loss",
        saveas=None, ax=None,
        **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize'))
    erange = np.arange(startidx, len(loss_hist_train))
    fplot = ax.loglog if log else ax.plot
    fplot(loss_hist_train[erange], loss_hist_valid[erange], 'k.-')
    fplot(loss_hist_train[erange[0]], loss_hist_valid[erange[0]], 'bo', 
          label=f'epoch {erange[0]}')
    fplot(loss_hist_train[erange[-1]], loss_hist_valid[erange[-1]], 'ro', 
          label=f'epoch {erange[-1]}')
    ax.set_xlabel(f"Training loss")
    ax.set_ylabel(f"Validation loss")
    ax.set_title(title)
    ax.legend()
    if saveas: plt.savefig(saveas)
    return ax


def plot_sigma_history(
        sigma_history, 
        log=False, 
        color='k',
        linestyle='-',
        linewidth=2,
        marker='.',
        title="Inferred noise parameter", 
        saveas=None,
        ax=None,
        sigma_true=None,
        sigma_true_legend_label=None,
        figsize=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    fplot = ax.semilogy if log else ax.plot
    fplot(
        sigma_history, 
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
    )
    if sigma_true is not None:
        if sigma_true_legend_label is None:
            sigma_true_legend_label = f'True $\\sigma={sigma_true:.3g}$'
        xlims = ax.get_xlim()
        ax.hlines(
            sigma_true, *xlims, 
            label=sigma_true_legend_label, 
            color='k', linestyle=':', alpha=0.5
        )          
        ax.set_xlim(*xlims)
        ax.legend()
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"inferred $\\sigma$")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax


def plot_learning_rate_history(
        lr_history, 
        log=False, 
        title="Learning Rate", 
        saveas=None,
        ax=None,
        color=None,
        linestyle=None,
        marker=None,
        linewidth=2,
        figsize=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    fplot = ax.semilogy if log else ax.plot
    fplot(
        lr_history, 
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
    )
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"learning rate")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax


def plot_dt_history(
        dt_history, 
        log=False, 
        color='k',
        linestyle='-',
        linewidth=2,
        marker='.',
        title="dt history", 
        saveas=None,
        ax=None,
        **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize'))
    fplot = ax.semilogy if log else ax.plot
    fplot(
        dt_history, 
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
    )
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"dt")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax


def plot_phi_inferred_vs_true(
        model, signal, landscape, 
        saveas=None,
        ax1_title="",
        ax2_title="",
        axes=None,
        figsize=(12,5),
        plot_radius=4,
        plot_res=200,
        lognormalize=True,
):  
    if axes is None:
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = axes

    model.plot_phi(
        signal=signal, 
        r=plot_radius, res=plot_res, plot3d=False,
        normalize=True, lognormalize=True,
        show=True,
        equal_axes=True,
        ax=ax1,
        title=ax1_title,
    )

    if landscape == 'phi1':
        f = func_phi1
    elif landscape == 'phi2':
        f = func_phi2
    else:
        print(landscape)
        raise RuntimeError()
    
    plot_landscape(
        f, r=plot_radius, res=plot_res, params=[0, 0], 
        lognormalize=lognormalize,
        clip=None,
        cbar_title="$\\log\\phi^*$" if lognormalize else "$\\phi^*$",
        saveas=None,
        ax=ax2,
        title=ax2_title,
    )

    if saveas: plt.savefig(saveas)
    return axes


def plot_neural_network(
        weights, biases,
        figsize=None,
        ax=None,
        normalization='layer',
        colormap='coolwarm_r',
        highlight_above=0.9,
        alpha=0.5,
):
    """ChatGPT Generated"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal')

    layers = []
    for w in weights:
        layers.append(w.shape[1])
    layers.append(weights[-1].shape[0])

    cmap = plt.get_cmap(colormap)
    if normalization == 'layer':
        norm = plt.Normalize(vmin=-1, vmax=1)
    else:
        maxabsval = np.max([np.abs(ws).max() for ws in weights])
        norm = plt.Normalize(vmin=-maxabsval, vmax=maxabsval)

    v_spacing = 1.0 / max(layers)
    h_spacing = 1.0 / (len(layers) - 1)

    # Draw nodes
    for layer_idx, layer_size in enumerate(layers):
        xpos = layer_idx * h_spacing
        for j in range(layer_size):
            if layer_size % 2 == 0:
                ypos = 0.5 + (0.5 + j - layer_size / 2) * v_spacing
            else:
                ypos = 0.5 + (j - layer_size // 2) * v_spacing
            circle = plt.Circle(
                (xpos, ypos), 
                v_spacing / 4.0, 
                color='w', 
                ec='k', 
                zorder=4
            )
            ax.add_artist(circle)
    
    # Draw edges
    for i, (layer_size_a, layer_size_b) in enumerate(
        zip(layers[:-1], layers[1:])
    ):
        xpos0 = i * h_spacing
        xpos1 = (i + 1) * h_spacing
        ws = weights[i]
        maxabsval = np.max(np.abs(ws))
        for a in range(layer_size_a):
            if layer_size_a % 2 == 0:
                ypos0 = 0.5 + (0.5 + a - layer_size_a / 2) * v_spacing
            else:
                ypos0 = 0.5 + (a - layer_size_a // 2) * v_spacing
            for b in range(layer_size_b):
                if layer_size_b % 2 == 0:
                    ypos1 = 0.5 + (0.5 + b - layer_size_b / 2) * v_spacing
                else:
                    ypos1 = 0.5 + (b - layer_size_b // 2) * v_spacing
                w = ws[b, a]
                # w_normed = (w - ws.min()) / (ws.max() - ws.min())
                if normalization == 'layer':
                    w_normed = w / maxabsval
                    col = cmap(w_normed)
                else:
                    w_normed = w
                    col = cmap(norm(w_normed))
                line = plt.Line2D(
                    [xpos0, xpos1],
                    [ypos0, ypos1],
                    lw=1.5, 
                    alpha=0.01 if np.abs(w_normed) < highlight_above else alpha,
                    color=col,
                )
                ax.add_artist(line)
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.get_figure().colorbar(sm, ax=ax)
    return ax

def _act_func_to_string(act_func):
    return 'softplus'
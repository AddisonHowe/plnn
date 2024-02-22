"""Plotting functions

"""

import numpy as np
import matplotlib.pyplot as plt

from .plot_landscapes import func_phi1, func_phi2, plot_landscape


def plot_training_loss_history(
        loss_hist_train, startidx=0, 
        log=False, title="Training Loss", 
        saveas=None, ax=None, **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    erange = np.arange(startidx, len(loss_hist_train))
    fplot = ax.semilogy if log else ax.plot
    fplot(1 + erange, loss_hist_train[erange], '.-', color=kwargs.get('color'))
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax


def plot_validation_loss_history(
        loss_hist_valid, startidx=0, 
        log=False, title="Validation Loss", 
        saveas=None, ax=None, **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    erange = np.arange(startidx, len(loss_hist_valid))
    fplot = ax.semilogy if log else ax.plot
    fplot(1 + erange, loss_hist_valid[erange], '.-', color=kwargs.get('color'))
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
        alpha_train=0.9,
        alpha_valid=0.5,
        saveas=None,
        ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    erange = np.arange(startidx, len(loss_hist_train))
    fplot = ax.semilogy if log else ax.plot
    fplot(1 + erange, loss_hist_train[erange], 'r.-', 
          label="Training", alpha=alpha_train)
    fplot(1 + erange, loss_hist_valid[erange], 'b.-', 
          label="Validation", alpha=alpha_valid)
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
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
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
        title="Inferred noise parameter", 
        saveas=None,
        ax=None,
        sigma_true=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    fplot = ax.semilogy if log else ax.plot
    fplot(sigma_history, 'k.-')
    if sigma_true is not None:
        xlims = ax.get_xlim()
        ax.hlines(
            sigma_true, *xlims, 
            label=f'$\\sigma={sigma_true:.3g}$',
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
        ax=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    fplot = ax.semilogy if log else ax.plot
    fplot(lr_history, 'k.-')
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"learning rate")
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
        normalize=True, log_normalize=True,
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

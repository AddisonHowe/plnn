"""Model Training Script
"""

import os
import time
import json
import numpy as np
import jax.random as jrandom
import equinox as eqx

from plnn.models import save_model


def train_model(
    model, 
    loss_fn, 
    optimizer,
    train_dataloader, 
    validation_dataloader,
    key,
    num_epochs=50,
    batch_size=1,
    hyperparams={},
    **kwargs
):
    """Train a PLNN model.

    Args:
        model (PLNN): Model to train.
        loss_fn (callable): Loss function.
        optimizer (optax.GradientTransformation): Optimizer.
        train_dataloader (DataLoader): Training dataloader.
        validation_dataloader (DataLoader): Validation dataloader.
        key (PRNGKey): Random number generator key.
        num_epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 1.
        hyperparams (dict, optional): Hyperparameters. Defaults to {}.
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    model_name = kwargs.get('model_name', 'model')
    outdir = kwargs.get('outdir', 'out')
    save_all = kwargs.get('save_all', False)
    plotting = kwargs.get('plotting', False)
    plotting_opts = kwargs.get('plotting_opts', {})
    report_every = kwargs.get('report_every', 10)
    verbosity = kwargs.get('verbosity', 1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    time0 = time.time()

    best_vloss = 1_000_000
    loss_hist_train = []
    loss_hist_valid = []
    sigma_hist = []

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Initial state plot, if specified
    if plotting:
        make_plots(0, model, outdir, plotting_opts)

    if verbosity: print(f"\nTraining model...\n")
    
    # Save initial model state
    os.makedirs(f"{outdir}/states", exist_ok=True)
    model_path = f"{outdir}/states/{model_name}_0.pth"
    if verbosity: print(f"Saving initial model state to: {model_path}")
    save_model(model_path, model, hyperparams)

    for epoch in range(num_epochs):
        if verbosity: print(f'EPOCH {epoch + 1}/{num_epochs}:', flush=True)
        etime0 = time.time()
        key, trainkey, validkey = jrandom.split(key, 3)

        # Training pass
        model, avg_tloss, opt_state = train_one_epoch(
            epoch, 
            model, 
            loss_fn, 
            optimizer,
            opt_state,
            train_dataloader,
            trainkey,
            batch_size=batch_size,
            report_every=report_every,
            verbosity=verbosity,
        )

        # Validation pass
        avg_vloss = validate_post_epoch(
            model,
            validation_dataloader,
            loss_fn, 
            validkey,
        )
        
        if verbosity:
            print(f"\tLOSS [training: {avg_tloss} | validation: {avg_vloss}]")
            print(f"\tTIME [epoch: {time.time() - etime0:.3g} sec]", flush=True)
        
        loss_hist_train.append(avg_tloss)
        loss_hist_valid.append(avg_vloss)
        np.save(f"{outdir}/training_loss_history.npy", loss_hist_train)
        np.save(f"{outdir}/validation_loss_history.npy", loss_hist_valid)

        sigma_hist.append(model.get_sigma())
        np.save(f"{outdir}/sigma_history.npy", sigma_hist)

        # Save the model's state
        if avg_vloss < best_vloss or save_all:
            model_path = f"{outdir}/states/{model_name}_{epoch + 1}.pth"
            if verbosity: print(f"\tSaving model to: {model_path}")
            save_model(model_path, model, hyperparams)

        # Track best performance
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            if verbosity: print(f"\tModel improved!!!", flush=True)
        
        # Plotting, if specified
        if plotting:
            make_plots(epoch + 1, model, outdir, plotting_opts)
        
    time1 = time.time()
    if verbosity: print(f"Finished training in {time1-time0:.3f} seconds.")
    return model


def train_one_epoch(
        epoch_idx, 
        model, 
        loss_fn, 
        optimizer,
        opt_state,
        dataloader, 
        key,
        batch_size=1,
        report_every=10,  # print running loss every 10 batches.
        verbosity=1,  # 0: none. 1: default. 2: debug.
    ):
    """One epoch of training.

    Args:
        epoch_idx (int): ...
        model (PLNN): ...
        dt (float): ...
        loss_fn (callable): ...
        optimizer (callable): ...
        opt_state (TODO): ...
        dataloader (DataLoader): ...
        batch_size (int): Default 1.
        device (str): Default 'cpu'.

    Returns:
        PLNN: updated model.
        float: average training loss across all batches.
        PyTree: optimizer state
    """

    epoch_running_loss = 0.
    batch_running_loss = 0.
    n = len(dataloader)
    if report_every <= 0 or report_every > n:
        report_every = n

    # Train over batches
    if verbosity:
        print("\tTraining over batches...")
    for bidx, data in enumerate(dataloader):
        inputs, y1 = data
        key, subkey = jrandom.split(key, 2)
        loss, model, opt_state = make_step(
            model, inputs, y1, optimizer, opt_state, loss_fn, subkey
        )
        
        epoch_running_loss += loss.item()
        batch_running_loss += loss.item()
        if bidx % report_every == (report_every - 1):
            avg_batch_loss = batch_running_loss / report_every
            if verbosity: 
                msg = f"\t\t[batch {bidx+1}/{n}] avg loss: {avg_batch_loss}"
                print(msg, flush=True)
            batch_running_loss = 0.

    avg_epoch_loss = epoch_running_loss / n
    return model, avg_epoch_loss, opt_state


def validate_post_epoch(
        model,
        validation_dataloader,
        loss_fn,
        key,
    ):
    """TODO

    Args:
        TODO

    Returns:
        _type_: _description_
    """
    n = len(validation_dataloader)
    running_vloss = 0.0
    for i, data in enumerate(validation_dataloader):
        inputs, y1 = data
        key, subkey = jrandom.split(key, 2)
        loss = validation_step(model, inputs, y1, loss_fn, subkey)
        running_vloss += loss.item()
    
    avg_vloss = running_vloss / n
    return avg_vloss


@eqx.filter_value_and_grad
def compute_loss(model, x, y, loss_fn, key):
    t0, y0, t1, sigparams = x
    y_pred = model(t0, t1, y0, sigparams, key)
    return loss_fn(y_pred, y)


@eqx.filter_jit
def make_step(model, x, y, optimizer, opt_state, loss_fn, key):
    loss, grads = compute_loss(model, x, y, loss_fn, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def validation_step(model, x, y, loss_fn, key):
    t0, y0, t1, sigparams = x
    y_pred = model(t0, t1, y0, sigparams, key)
    vloss = loss_fn(y_pred, y)
    return vloss
    

def make_plots(epoch, model, outdir, plotting_opts):
    """Make plots at the end of each epoch.
    
    Args:
        plotting_opts (dict) : dictionary of options. Handles following keys:
            ...
    """
    plot_radius = plotting_opts.get('plot_radius', 4)
    plot_res = plotting_opts.get('plot_res', 50)
    plot_phi_heatmap = plotting_opts.get('plot_phi_heatmap', False)
    plot_phi_landscape = plotting_opts.get('plot_phi_landscape', False)
    plot_phi_heatmap_norm = plotting_opts.get('plot_phi_heatmap_norm', False)
    plot_phi_landscape_norm = plotting_opts.get('plot_phi_landscape_norm', False)
    plot_phi_heatmap_lognorm = plotting_opts.get('plot_phi_heatmap_lognorm', True)
    plot_phi_landscape_lognorm = plotting_opts.get('plot_phi_landscape_lognorm', False)

    if plot_phi_heatmap: 
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=False, log_normalize=False,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/phi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=False, log_normalize=False,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/phi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, log_normalize=False,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/normphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, log_normalize=False,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/normphi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, log_normalize=True,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/logphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, log_normalize=True,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/logphi_landscape_{epoch}.png",
        )

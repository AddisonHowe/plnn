"""Model Training Script
"""

import time
import json
import numpy as np
import jax.random as jrandom
import equinox as eqx

from plnn.models import PLNN, make_model


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
    plotting = kwargs.get('plotting', False)
    plotting_opts = kwargs.get('plotting_opts', {})
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

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}:', flush=True)
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
            verbosity=verbosity,
        )

        # Validation pass
        avg_vloss = validate_post_epoch(
            model,
            validation_dataloader,
            loss_fn, 
            validkey,
        )
        
        print("LOSS [train: {}] [valid: {}] TIME [epoch: {:.3g} sec]".format(
            avg_tloss, avg_vloss, time.time() - etime0), flush=True)
        
        loss_hist_train.append(avg_tloss)
        loss_hist_valid.append(avg_vloss)
        np.save(f"{outdir}/training_loss_history.npy", loss_hist_train)
        np.save(f"{outdir}/validation_loss_history.npy", loss_hist_valid)

        sigma_hist.append(model.get_sigma())
        np.save(f"{outdir}/sigma_history.npy", sigma_hist)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"{outdir}/{model_name}_{epoch}.pth"
            print("Saving model.")
            save(model_path, model, hyperparams)
        
        # Plotting, if specified
        if plotting:
            make_plots(epoch + 1, model, outdir, plotting_opts)
        
    time1 = time.time()
    print(f"Finished training in {time1-time0:.3f} seconds.")
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
        report_every=100,  # print running loss every 10 batches.
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
        float: training loss with respect to the last batch.
        PyTree: optimizer state
    """

    running_loss = 0.
    last_loss = 0.
    if report_every <= 0 or report_every > len(dataloader):
        report_every = len(dataloader)

    # Train over batches
    for bidx, data in enumerate(dataloader):
        inputs, y1 = data
        key, subkey = jrandom.split(key, 2)
        loss, model, opt_state = make_step(
            model, inputs, y1, optimizer, opt_state, loss_fn, subkey
        )
        
        running_loss += loss.item()
        if bidx % report_every == (report_every - 1):
            last_loss = running_loss / report_every  # average loss per batch
            if verbosity: 
                print(f'\tbatch {bidx + 1} loss: {last_loss}', flush=True)
            running_loss = 0.

    return model, last_loss, opt_state


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
    running_vloss = 0.0
    for i, data in enumerate(validation_dataloader):
        inputs, y1 = data
        key, subkey = jrandom.split(key, 2)
        loss = validation_step(model, inputs, y1, loss_fn, subkey)
        running_vloss += loss.item()
    
    avg_vloss = running_vloss / (i + 1)
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



def save(filename, model, hyperparams={}):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model, _ = make_model(key=jrandom.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)
    

def make_plots(epoch, model, outdir, plotting_opts):
    """Make plots at the end of each epoch.
    
    Args:
        plotting_opts (dict) : dictionary of options. Handles following keys:
            ...
    """
    plot_radius = plotting_opts.get('plot_radius', 4)
    plot_res = plotting_opts.get('plot_res', 50)
    plot_phi_heatmap = plotting_opts.get('plot_phi_heatmap', True)
    plot_phi_landscape = plotting_opts.get('plot_phi_landscape', False)
    plot_phi_heatmap_norm = plotting_opts.get('plot_phi_heatmap_norm', False)
    plot_phi_landscape_norm = plotting_opts.get('plot_phi_landscape_norm', False)
    plot_phi_heatmap_lognorm = plotting_opts.get('plot_phi_heatmap_lognorm', False)
    plot_phi_landscape_lognorm = plotting_opts.get('plot_phi_landscape_lognorm', False)

    if plot_phi_heatmap: 
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=False, log_normalize=False,
            title=f"$\\phi$ (Epoch {epoch})",
            saveas=f"{outdir}/images/phi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=False, log_normalize=False,
            title=f"$\\phi$ (Epoch {epoch})",
            saveas=f"{outdir}/images/phi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, log_normalize=False,
            title=f"$\\phi$ (Epoch {epoch})",
            saveas=f"{outdir}/images/normphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, log_normalize=False,
            title=f"$\\phi$ (Epoch {epoch})",
            saveas=f"{outdir}/images/normphi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, log_normalize=True,
            title=f"$\\phi$ (Epoch {epoch})",
            saveas=f"{outdir}/images/logphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, log_normalize=True,
            title=f"$\\phi$ (Epoch {epoch})",
            saveas=f"{outdir}/images/logphi_landscape_{epoch}.png",
        )

"""Model Training Script

"""

import os
import time
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
from optax import skip_not_finite
import matplotlib.pyplot as plt

from plnn.models.plnn import PLNN
from plnn.pl.plotting import plot_loss_history

def train_model(
    model, 
    loss_fn, 
    optimizer,
    train_dataloader, 
    validation_dataloader,
    key,
    num_epochs=50,
    batch_size=1,
    fix_noise=False,
    hyperparams={},
    reduce_dt_on_nan=False, 
    dt_reduction_factor=0.5, 
    reduce_cf_on_nan=False, 
    cf_reduction_factor=0.5, 
    nan_max_attempts=4,
    **kwargs
) -> PLNN:
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
        fix_noise (boolean, optional): Whether to fix the noise parameter.
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
    logprint = kwargs.get('logprint', None)
    error_raiser = kwargs.get('error_raiser', None)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if logprint is None:
        def logprint(s, flush=True):
            print(s, flush=flush)

    if error_raiser is None:
        def log_and_raise_runtime_error(msg):
            raise RuntimeError(msg)
    else:
        log_and_raise_runtime_error = error_raiser


    time0 = time.time()

    best_vloss = 1_000_000
    loss_hist_train = []
    loss_hist_valid = []
    learn_rate_hist = []
    sigma_hist = []
    tilt_weights = []
    tilt_bias = []

    if fix_noise:
        filter_spec = jtu.tree_map(lambda _: True, model)
        filter_spec = eqx.tree_at(
            lambda m: m.logsigma, filter_spec, 
            replace=False
        )
    else:
        filter_spec = None

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Initial state plot, if specified
    if plotting:
        make_plots(0, model, outdir, plotting_opts)

    if verbosity: logprint(f"\nTraining model...\n")
    
    # Save initial model state
    os.makedirs(f"{outdir}/states", exist_ok=True)
    model_path = f"{outdir}/states/{model_name}_0.pth"
    if verbosity: logprint(f"Saving initial model state to: {model_path}")
    model.save(model_path, hyperparams)

    for epoch in range(num_epochs):
        if verbosity: logprint(f'EPOCH {epoch + 1}/{num_epochs}:')
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
            logprint=logprint,
            error_raiser=error_raiser,
            fix_noise=fix_noise,
            filter_spec=filter_spec,
            reduce_dt_on_nan=reduce_dt_on_nan,
            dt_reduction_factor=dt_reduction_factor,
            reduce_cf_on_nan=reduce_cf_on_nan,
            cf_reduction_factor=cf_reduction_factor,
            nan_max_attempts=nan_max_attempts,
            hyperparams=hyperparams,
            model_name=model_name,
            outdir=outdir,
        )

        if np.isnan(avg_tloss):
            msg = f"nan encountered in epoch {epoch} (training loss)."
            log_and_raise_runtime_error(msg)

        # Validation pass
        avg_vloss = validate_post_epoch(
            model,
            validation_dataloader,
            loss_fn, 
            validkey,
        )

        if np.isnan(avg_vloss):
            pass
            msg = f"nan encountered in epoch {epoch} (validation loss)."
            logprint(msg)
        
        if hasattr(opt_state[1], 'hyperparams'):
            lr = opt_state[1].hyperparams['learning_rate']
        else:
            lr = 0

        if verbosity:
            if lr: 
                logprint(f"\tLearning Rate: {lr:.6g}")
            logprint(f"\tLOSS [training: {avg_tloss} | validation: {avg_vloss}]")
            logprint(f"\tTIME [epoch: {time.time() - etime0:.3g} sec]")
        
        loss_hist_train.append(avg_tloss)
        loss_hist_valid.append(avg_vloss)
        learn_rate_hist.append(lr)
        np.save(f"{outdir}/training_loss_history.npy", loss_hist_train)
        np.save(f"{outdir}/validation_loss_history.npy", loss_hist_valid)
        np.save(f"{outdir}/learning_rate_history.npy", learn_rate_hist)

        sigma_hist.append(model.get_sigma())
        np.save(f"{outdir}/sigma_history.npy", sigma_hist)

        tilt_weights.append(model.get_parameters()['tilt.w'])
        np.save(f"{outdir}/tilt_weights_history.npy", tilt_weights)

        tilt_bias.append(model.get_parameters()['tilt.b'])
        np.save(f"{outdir}/tilt_bias_history.npy", tilt_bias)

        # Save the model's state
        if avg_vloss < best_vloss or save_all:
            model_path = f"{outdir}/states/{model_name}_{epoch + 1}.pth"
            if verbosity: logprint(f"\tSaving model to: {model_path}")
            model.save(model_path, hyperparams)

        # Plotting, if specified
        if plotting and (avg_vloss < best_vloss or save_all):
            make_plots(
                epoch + 1, model, outdir, plotting_opts,
                plot_losses=True,
                loss_hist_train=loss_hist_train,
                loss_hist_valid=loss_hist_valid,
            )
            
        # Track best performance
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            if verbosity: logprint(f"\tModel improved!!!")
        
        
    time1 = time.time()
    if verbosity: logprint(f"Finished training in {time1-time0:.3f} seconds.")
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
        fix_noise=False,
        filter_spec=None,
        report_every=10,  # print running loss every 10 batches.
        verbosity=1,  # 0: none. 1: default. 2: debug.
        reduce_dt_on_nan=False, 
        dt_reduction_factor=0.5, 
        reduce_cf_on_nan=False, 
        cf_reduction_factor=0.5, 
        nan_max_attempts=4,
        hyperparams=None,
        model_name=None,
        outdir=None,
        **kwargs
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
        fix_noise (bool): ...

    Returns:
        PLNN: updated model.
        float: average training loss across all batches.
        PyTree: optimizer state
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    logprint = kwargs.get('logprint', None)
    error_raiser = kwargs.get('error_raiser', None)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if logprint is None:
        def logprint(s, flush=True):
            print(s, flush=flush)

    if error_raiser is None:
        def log_and_raise_runtime_error(msg):
            raise RuntimeError(msg)
    else:
        log_and_raise_runtime_error = error_raiser

    debug_dir = f"{outdir}/debug"

    epoch_running_loss = 0.
    batch_running_loss = 0.
    n = len(dataloader)
    if report_every <= 0 or report_every > n:
        report_every = n

    # Train over batches
    if verbosity:
        logprint("\tTraining over batches...")
    for bidx, data in enumerate(dataloader):
        inputs, y1 = data
        key, subkey = jrandom.split(key, 2)
        orig_model = model  # keep a copy of the original model at batch start
        orig_opt_state = opt_state  # keep a copy of the original opt state
        prev_model = model  # keep a copy of the previous model in case of error
        stepped = False
        attempts = 0
        while not stepped:
            if fix_noise:
                model, opt_state, loss = make_step_partitioned(
                    prev_model, inputs, y1, optimizer, orig_opt_state, loss_fn, subkey, 
                    filter_spec
                )
            else:
                model, opt_state, loss = make_step(
                    prev_model, inputs, y1, optimizer, orig_opt_state, loss_fn, subkey, 
                )

            if jnp.isfinite(loss.item()):
                stepped = True
            else:
                if attempts == 0:
                    os.makedirs(debug_dir, exist_ok=True)
                # Raise an error if the number of attempts reaches the limit.
                if attempts == nan_max_attempts:
                    msg = "\tEncountered nan in loss and reached the maximum "
                    msg += f"number of model alterations: {nan_max_attempts}."
                    log_and_raise_runtime_error(msg)
                    
                # Save the pre-step model
                model_path = f"{debug_dir}/{model_name}_{epoch_idx + 1}_{bidx}_err_prestep{attempts}.pth"
                prev_model.save(model_path, hyperparams)
                # Save the post-step model
                model_path = f"{debug_dir}/{model_name}_{epoch_idx + 1}_{bidx}_err_poststep{attempts}.pth"
                model.save(model_path, hyperparams)
                    
                # Perform model surgery
                msg = f"\tEncountered nan in loss. Reverting update and performing"
                msg += f" model surgery ({attempts + 1}/{nan_max_attempts})."
                if reduce_dt_on_nan:
                    where = lambda m: m.dt0
                    model = eqx.tree_at(
                        where, orig_model, 
                        prev_model.dt0 * dt_reduction_factor
                    )
                    hyperparams['dt0'] = model.dt0
                    prev_model = model
                    msg += f"\n\t\tNew model dt0: {model.dt0}"
                if reduce_cf_on_nan:
                    where = lambda m: m.confinement_factor
                    model = eqx.tree_at(
                        where, orig_model, 
                        prev_model.confinement_factor * cf_reduction_factor
                    )
                    hyperparams['confinement_factor'] = model.confinement_factor
                    prev_model = model
                    msg += f"\n\t\tNew model confinement_factor: {model.confinement_factor}"
                logprint(msg)
                
                # Save the resulting model
                model_path = f"{debug_dir}/{model_name}_{epoch_idx + 1}_{bidx}_postop{attempts}.pth"
                model.save(model_path, hyperparams)

                if jnp.any(jnp.isnan(model.get_parameters()['phi.w'][0])):
                    logprint("!!! Got nan in saved model phi.w[0] !!!")

                attempts += 1
        
        epoch_running_loss += loss.item()
        batch_running_loss += loss.item()
        if bidx % report_every == (report_every - 1):
            avg_batch_loss = batch_running_loss / report_every
            if verbosity: 
                msg = f"\t\t[batch {bidx+1}/{n}] avg loss: {avg_batch_loss}"
                if hasattr(opt_state[1], 'hyperparams'):
                    lr = opt_state[1].hyperparams['learning_rate']
                    msg += f"\t\t[learning rate: {lr:.5g}]"
                logprint(msg)
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
    return model, opt_state, loss


@eqx.filter_value_and_grad
def compute_loss_partitioned(diff_model, static_model, x, y, loss_fn, key):
    model = eqx.combine(diff_model, static_model)
    t0, y0, t1, sigparams = x
    y_pred = model(t0, t1, y0, sigparams, key)
    return loss_fn(y_pred, y)


@eqx.filter_jit
def make_step_partitioned(model, x, y, optimizer, opt_state, loss_fn, key, 
                          filter_spec):
    diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = compute_loss_partitioned(
        diff_model, static_model, x, y, loss_fn, key
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def validation_step(model, x, y, loss_fn, key):
    t0, y0, t1, sigparams = x
    y_pred = model(t0, t1, y0, sigparams, key)
    vloss = loss_fn(y_pred, y)
    return vloss
    

def make_plots(epoch, model, outdir, plotting_opts, **kwargs):
    """Make plots at the end of each epoch.
    
    Args:
        plotting_opts (dict) : dictionary of options. Handles following keys:
            ...
    """
    plot_radius = plotting_opts.get('plot_radius', 4)
    plot_res = plotting_opts.get('plot_res', 50)
    equal_axes = plotting_opts.get('equal_axes', True)
    plot_phi_heatmap = plotting_opts.get('plot_phi_heatmap', False)
    plot_phi_landscape = plotting_opts.get('plot_phi_landscape', False)
    plot_phi_heatmap_norm = plotting_opts.get('plot_phi_heatmap_norm', False)
    plot_phi_landscape_norm = plotting_opts.get('plot_phi_landscape_norm', False)
    plot_phi_heatmap_lognorm = plotting_opts.get('plot_phi_heatmap_lognorm', True)
    plot_phi_landscape_lognorm = plotting_opts.get('plot_phi_landscape_lognorm', False)
    plot_losses = kwargs.get('plot_losses', False)
    loss_hist_train = kwargs.get('loss_hist_train', None)
    loss_hist_valid = kwargs.get('loss_hist_valid', None)

    do_plot_loss_hist = plot_losses and (loss_hist_train is not None) \
                                    and (loss_hist_valid is not None)
    if do_plot_loss_hist:
        plot_loss_history(
            np.array(loss_hist_train),
            np.array(loss_hist_valid),
            startidx=0, 
            log=True, 
            title="Loss History", 
            alpha_train=0.9,
            alpha_valid=0.5,
            saveas=f"{outdir}/images/loss_history.png",
        )
        plt.close()
    if plot_phi_heatmap: 
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=False, log_normalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/phi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=False, log_normalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/phi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, log_normalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/normphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, log_normalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/normphi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, log_normalize=True,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/logphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, log_normalize=True,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/logphi_landscape_{epoch}.png",
        )

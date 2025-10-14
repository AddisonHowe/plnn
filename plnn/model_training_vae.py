"""Model Training Script for VAEPLNNs

"""

import os, sys, signal, time
import numpy as np
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from plnn.models import VAEPLNN
from plnn.pl.plotting import plot_loss_history, plot_sigma_history
from plnn.loss_functions import batched_l2_ensemble_loss, batched_kl_vae_loss

def train_model_vae(
    model, 
    loss_fn, 
    optimizer,
    train_dataloader, 
    validation_dataloader,
    key, *, 
    num_epochs=50,
    batch_size=1,
    patience=100,
    min_epochs=0,
    dt_schedule=None,
    fix_noise=False,
    hyperparams={},
    reduce_dt_on_nan=False, 
    dt_reduction_factor=0.5, 
    reduce_cf_on_nan=False, 
    cf_reduction_factor=0.5, 
    nan_max_attempts=4,
    **kwargs
) -> VAEPLNN:
    """Train a PLNN model.

    Args:
        model (VAEPLNN): Model to train.
        loss_fn (callable): Loss function.
        optimizer (optax.GradientTransformation): Optimizer.
        train_dataloader (DataLoader): Training dataloader.
        validation_dataloader (DataLoader): Validation dataloader.
        key (PRNGKey): Random number generator key.
        num_epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 1.
        patience (int, optional): Patience. Defaults to 100.
        fix_noise (boolean, optional): Whether to fix the noise parameter.
        hyperparams (dict, optional): Hyperparameters. Defaults to {}.
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    model_name = kwargs.get('model_name', 'model')
    outdir = kwargs.get('outdir', 'out')
    save_all = kwargs.get('save_all', False)
    save_every = kwargs.get('save_every', 100)
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
    best_epoch = 0

    loss_hist_train = []
    loss_hist_valid = []
    learn_rate_hist = []
    sigma_hist = []
    tilt_weights = []
    tilt_bias = []
    dt_hist = []

    def sigterm_handler(signalnum, handler):
        logprint('*** RECEIVED SIGTERM *** Raising KeyboardInterrupt!')
        np.save(f"{outdir}/training_loss_history.npy", loss_hist_train)
        np.save(f"{outdir}/validation_loss_history.npy", loss_hist_valid)
        np.save(f"{outdir}/learning_rate_history.npy", learn_rate_hist)
        np.save(f"{outdir}/sigma_history.npy", sigma_hist)
        np.save(f"{outdir}/tilt_weights_history.npy", tilt_weights)
        np.save(f"{outdir}/tilt_bias_history.npy", tilt_bias)
        np.save(f"{outdir}/dt_hist.npy", dt_hist)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, sigterm_handler)

    if fix_noise:
        filter_spec = jtu.tree_map(lambda _: True, model)
        filter_spec = eqx.tree_at(
            lambda m: m.logsigma, filter_spec, 
            replace=False
        )
    else:
        filter_spec = None

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    dt_current = model.get_dt0()

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

        # Compute model dt value from scheduler, and update if needed.
        if dt_schedule:
            dt_new = dt_schedule(epoch)
            if dt_current != dt_new:
                model = eqx.tree_at(lambda m: m.dt0, model, dt_new)
                dt_current = dt_new
                assert dt_current == model.get_dt0(), "Model dt0 does not match"
                np.save(f"{outdir}/dt_hist.npy", dt_hist)
        dt_hist.append(dt_current)

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
            msg = f"nan encountered in epoch {epoch + 1} (training loss)."
            log_and_raise_runtime_error(msg)

        # Validation pass
        avg_vloss = validate_post_epoch(
            model,
            validation_dataloader,
            loss_fn, 
            validkey,
        )

        if np.isnan(avg_vloss):
            msg = f"nan encountered in epoch {epoch + 1} (validation loss)."
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
        sigma_hist.append(model.get_sigma())
        tilt_weights.append(model.get_parameters()['tilt.w'])
        tilt_bias.append(model.get_parameters()['tilt.b'])

        np.save(f"{outdir}/training_loss_history.npy", loss_hist_train)
        np.save(f"{outdir}/validation_loss_history.npy", loss_hist_valid)
        np.save(f"{outdir}/learning_rate_history.npy", learn_rate_hist)
        np.save(f"{outdir}/sigma_history.npy", sigma_hist)
        np.save(f"{outdir}/tilt_weights_history.npy", tilt_weights)
        np.save(f"{outdir}/tilt_bias_history.npy", tilt_bias)

        model_improved = avg_vloss < best_vloss
        exceeded_patience = (epoch - best_epoch + 1 > patience) and epoch > min_epochs

        # Save the model's state
        if model_improved or save_all or exceeded_patience:
            model_path = f"{outdir}/states/{model_name}_{epoch + 1}.pth"
            if verbosity: logprint(f"\tSaving model to: {model_path}")
            model.save(model_path, hyperparams)

        # Plotting, if specified
        if plotting and (
            model_improved or save_all or ((epoch + 1) % save_every == 0) \
                or exceeded_patience
        ):
            make_plots(
                epoch + 1, model, outdir, plotting_opts,
                loss_hist_train=loss_hist_train,
                loss_hist_valid=loss_hist_valid,
                sigma_hist=sigma_hist,
            )
            
        # Track best performance
        if model_improved:
            best_vloss = avg_vloss
            best_epoch = epoch + 1
            if verbosity: logprint(f"\tModel improved!!!")
        
        # Early stopping
        if exceeded_patience:
            time1 = time.time()
            msg = f"Halted early. No improvement in validation loss " + \
                f"for {patience} epochs.\n" + \
                f"Finished training in {time1-time0:.3f} seconds."
            if verbosity: 
                logprint(msg)
            np.save(f"{outdir}/dt_hist.npy", dt_hist)
            return model
        
    np.save(f"{outdir}/dt_hist.npy", dt_hist)
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

            if jnp.isnan(loss.item()):
                model, prev_model = handle_nan_loss(
                    attempts,
                    model=model,
                    prev_model=prev_model, 
                    orig_model=orig_model,
                    nan_max_attempts=nan_max_attempts,
                    debug_dir=debug_dir, 
                    model_name=model_name,
                    epoch_idx=epoch_idx,
                    batch_idx=bidx,
                    hyperparams=hyperparams,
                    reduce_dt_on_nan=reduce_dt_on_nan,
                    dt_reduction_factor=dt_reduction_factor,
                    reduce_cf_on_nan=reduce_cf_on_nan,
                    cf_reduction_factor=cf_reduction_factor,
                    logprint=logprint,
                    log_and_raise_runtime_error=log_and_raise_runtime_error,
                )
                attempts += 1
            else:
                stepped = True
                
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
        
        if isinstance(model, VAEPLNN) and jnp.any(
                jnp.isnan(model.get_parameters()['phi.w'][0])):
            msg = "!!! UPDATED MODEL HAS NAN VALUES IN PHI.W[0] !!!"
            log_and_raise_runtime_error(msg)

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
    y_pred, res = model(t0, t1, y0, sigparams, key, return_all=True)
    z0_mu = res["z0_mu"]
    z0_logvar = res["z0_logvar"]
    y0hat = res["y0hat"]
    # Sum distributional loss, reconstruction loss, and KL divergence loss
    dist_loss = loss_fn(y_pred, y)
    rec_loss = batched_l2_ensemble_loss(y0hat, y0)
    kl_loss = batched_kl_vae_loss(z0_mu, z0_logvar)
    loss = dist_loss + rec_loss + kl_loss
    return loss


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
    y_pred, res = model(t0, t1, y0, sigparams, key, return_all=True)
    z0_mu = res["z0_mu"]
    z0_logvar = res["z0_logvar"]
    y0hat = res["y0hat"]
    # Sum distributional loss, reconstruction loss, and KL divergence loss
    dist_loss = loss_fn(y_pred, y)
    rec_loss = batched_l2_ensemble_loss(y0hat, y0)
    kl_loss = batched_kl_vae_loss(z0_mu, z0_logvar)
    loss = dist_loss + rec_loss + kl_loss
    return loss


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
    y_pred, res = model(t0, t1, y0, sigparams, key, return_all=True)
    z0_mu = res["z0_mu"]
    z0_logvar = res["z0_logvar"]
    y0hat = res["y0hat"]
    # Sum distributional loss, reconstruction loss, and KL divergence loss
    dist_loss = loss_fn(y_pred, y)
    rec_loss = batched_l2_ensemble_loss(y0hat, y0)
    kl_loss = batched_kl_vae_loss(z0_mu, z0_logvar)
    loss = dist_loss + rec_loss + kl_loss
    return loss


def _sample_y(y, ncells, key):
        nbatches, _, dim = y.shape
        y_samp = jnp.empty([nbatches, ncells, dim])
        if y.shape[1] < ncells:
            # Sample with replacement
            for bidx in range(y.shape[0]):
                key, subkey = jrandom.split(key, 2)
                samp_idxs = jnp.array(
                    jrandom.choice(subkey, y.shape[1], (ncells,), True),
                    dtype=int,
                )
                y_samp = y_samp.at[bidx,:].set(y[bidx,samp_idxs])
        else:
            # Sample without replacement
            for bidx in range(y.shape[0]):
                key, subkey = jrandom.split(key, 2)
                samp_idxs = jnp.array(
                    jrandom.choice(subkey, y.shape[1], (ncells,), False),
                    dtype=int,
                )
                y_samp = y_samp.at[bidx,:].set(y[bidx,samp_idxs])
        return y_samp


def get_subsample(batched_inputs, batched_y1s, ncells, key):
    batched_t0s = batched_inputs[0]
    batched_y0s = batched_inputs[1]
    batched_t1s = batched_inputs[2]
    batched_sigparams = batched_inputs[3]
    key, subkey1, subkey2 = jrandom.split(key, 3)
    y0_samps = _sample_y(batched_y0s, ncells, subkey1)
    y1_samps = _sample_y(batched_y1s, ncells, subkey2)
    inputs = (batched_t0s, y0_samps, batched_t1s, batched_sigparams)
    outputs = y1_samps
    return inputs, outputs


def handle_nan_loss(
        attempts,
        model, 
        prev_model, 
        orig_model,
        nan_max_attempts,
        debug_dir, 
        model_name,
        epoch_idx,
        batch_idx,
        hyperparams,
        reduce_dt_on_nan,
        dt_reduction_factor,
        reduce_cf_on_nan,
        cf_reduction_factor,
        logprint,
        log_and_raise_runtime_error,
):
    if attempts == 0:
        os.makedirs(debug_dir, exist_ok=True)
    # Raise an error if the number of attempts reaches the limit.
    if attempts == nan_max_attempts:
        msg = "\tEncountered nan in loss and reached the maximum "
        msg += f"number of model alterations: {nan_max_attempts}."
        log_and_raise_runtime_error(msg)
        
    # Save the pre-step model
    model_path = "{}/{}_{}_{}_err_prestep{}.pth".format(
        debug_dir, model_name, epoch_idx + 1, batch_idx, attempts)
    prev_model.save(model_path, hyperparams)
    # Save the post-step model
    model_path = "{}/{}_{}_{}_err_poststep{}.pth".format(
        debug_dir, model_name, epoch_idx + 1, batch_idx, attempts)
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
    model_path = "{}/{}_{}_{}_postop{}.pth".format(
        debug_dir, model_name, epoch_idx + 1, batch_idx, attempts)
    model.save(model_path, hyperparams)

    if jnp.any(jnp.isnan(model.get_parameters()['phi.w'][0])):
        logprint("!!! Got nan in saved model phi.w[0] !!!")

    return model, prev_model
    

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
    plot_losses = plotting_opts.get('plot_losses', False)
    loss_hist_train = kwargs.get('loss_hist_train', None)
    loss_hist_valid = kwargs.get('loss_hist_valid', None)
    plot_sigma_hist = plotting_opts.get('plot_sigma_hist', False)
    sigma_true = plotting_opts.get('sigma_true', None)
    sigma_hist = kwargs.get('sigma_hist', None)

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

    if plot_sigma_hist and (sigma_hist is not None):
        plot_sigma_history(
            np.array(sigma_hist),
            log=True, 
            sigma_true=sigma_true,
            title="$\\sigma$ history", 
            saveas=f"{outdir}/images/sigma_history.png",
        )
        plt.close()
        
    if plot_phi_heatmap: 
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=False, lognormalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/phi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=False, lognormalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/phi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, lognormalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/normphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_norm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, lognormalize=False,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/normphi_landscape_{epoch}.png",
        )
    if plot_phi_heatmap_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=False,
            normalize=True, lognormalize=True,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/logphi_heatmap_{epoch}.png",
        )
    if plot_phi_landscape_lognorm:
        model.plot_phi(
            r=plot_radius, res=plot_res, plot3d=True,
            normalize=True, lognormalize=True,
            equal_axes=equal_axes,
            title=f"$\\phi$ ({epoch} epochs)",
            saveas=f"{outdir}/images/logphi_landscape_{epoch}.png",
        )

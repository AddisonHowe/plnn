"""Model Training Script
"""

import sys, time
import json
import numpy as np
import jax.random as jrandom
import equinox as eqx

from plnn.models import PLNN


def train_model(
        model, 
        loss_fn, 
        optimizer,
        train_dataloader, 
        validation_dataloader,
        key,
        num_epochs=50,
        batch_size=1,
        **kwargs
    ):
    """TODO

    Args:
        model (_type_): _description_
        dt (_type_): _description_
        loss_fn (_type_): _description_
        optimizer (_type_): _description_
        train_dataloader (_type_): _description_
        validation_dataloader (_type_): _description_
        key (_type_): _description_
        num_epochs (int, optional): _description_. Defaults to 50.
        batch_size (int, optional): _description_. Defaults to 1.
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
    print(opt_state)

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
            hyperparams = model.get_hyperparameters()
            save(model_path, hyperparams, model)
        
        # Plotting, if specified
        if plotting:
            make_plots(epoch, model, outdir, plotting_opts)
        
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
        verbosity=1,   # 0: none. 1: default. 2: debug.
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
        float: last loss
    """

    running_loss = 0.
    last_loss = 0.

    # Train over batches
    for i, data in enumerate(dataloader):
        inputs, y1 = data
        key, subkey = jrandom.split(key, 2)
        loss, model, opt_state = make_step(
            model, inputs, y1, optimizer, opt_state, loss_fn, subkey
        )
        
        running_loss += loss.item()
        if i % batch_size == (batch_size - 1):
            last_loss = running_loss / batch_size  # loss per batch
            if verbosity: 
                print(f'\tbatch {i + 1} loss: {last_loss}', flush=True)
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


def make(*, key, ncells):
    return PLNN(
        ndim=2,
        nsig=2,
        ncells=ncells,
        key=key,
        nsigparams=5,
        sigma_init=1e-2,
    )


def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make(key=jrandom.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)
    

def make_plots():
    pass

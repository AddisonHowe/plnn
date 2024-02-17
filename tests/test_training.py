"""Tests of model training

"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from tests.conftest import DATDIR, TMPDIR, remove_dir

from plnn.models import DeepPhiPLNN
from plnn.dataset import get_dataloaders
from plnn.loss_functions import mean_cov_loss, kl_divergence_loss
from plnn.optimizers import get_optimizer_args, select_optimizer
from plnn.model_training import train_model


#####################
##  Configuration  ##
#####################

W1 = np.array([
    [1, 3],
    [2, 2],
    [3, 1],
], dtype=float)

W2 = np.array([
    [1, 1, -2],
    [0, 1, 0],
    [-1, 2, 1],
], dtype=float)

W3 = np.array([
    [2, 3, 1]
], dtype=float)

WT1 = np.array([
    [2, 4],
    [-1, 1],
], dtype=float)

def get_model(
        ws, wts, dtype, 
        sigma=0, 
        seed=None, 
        ncells=4, 
        dt=0.1, 
        sample_cells=False, 
        signal_type='binary', 
        nsigparams=3,
        confine=False,
        confinement_factor=1.0,
) -> DeepPhiPLNN:
    """Get an initialized model for testing purposes."""
    nprng = np.random.default_rng(seed)
    key_model = jrandom.PRNGKey(nprng.integers(2**32))
    key_init = jrandom.PRNGKey(nprng.integers(2**32))
    model = DeepPhiPLNN(
        key=key_model,
        dtype=dtype,
        ndims=2, 
        nparams=2, 
        nsigs=2, 
        signal_type=signal_type,
        nsigparams=nsigparams,
        ncells=ncells, 
        sigma_init=sigma,
        dt0=dt,
        confine=confine,
        confinement_factor=confinement_factor,
        include_phi_bias=False, 
        include_tilt_bias=False, 
        phi_hidden_dims=[3,3],
        phi_hidden_acts='tanh',
        phi_final_act=None,
        phi_layer_normalize=False,
        tilt_hidden_dims=[],
        tilt_hidden_acts=None,
        tilt_final_act=None,
        tilt_layer_normalize=False,
        sample_cells=sample_cells,
        solver='euler'
    )
    model = model.initialize(
        key_init,
        dtype=dtype,
        init_phi_weights_method='explicit',
        init_phi_weights_args=[ws],
        init_phi_bias_method='none',
        init_phi_bias_args=[],
        init_tilt_weights_method='explicit',
        init_tilt_weights_args=[wts],
        init_tilt_bias_method='none',
        init_tilt_bias_args=[],
    )
    return model
        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################


@pytest.mark.parametrize('dtype', [jnp.float32])
@pytest.mark.parametrize('datdir_train, datdir_valid', [
    [f"{DATDIR}/simtest1/data_train", f"{DATDIR}/simtest1/data_valid"],
])
@pytest.mark.parametrize('sigma', [0.3]) # [0.3, 0.5]
@pytest.mark.parametrize('loss_fn', [kl_divergence_loss, mean_cov_loss])
@pytest.mark.parametrize('opt_method', ['sgd']) # ['sgd', 'adam', 'rms']
def test_fixed_sigma_training(dtype, datdir_train, datdir_valid, 
                              sigma, loss_fn, opt_method):
    nprng = np.random.default_rng()

    train_dataloader, valid_dataloader, train_dset, _ = get_dataloaders(
        datdir_train, datdir_valid, 4, 4, 
        batch_size_train=1, batch_size_valid=1, 
        ndims=2, dtype=dtype, return_datasets=True,
    )

    model = get_model([W1, W2, W3], [WT1], dtype, sigma=sigma, ncells=10)
    
    optimizer_args = {
        'lr_schedule' : 'constant',
        'learning_rate' : 0.01,
        'momentum' : 0.9,
        'weight_decay' : 0.5, 
        'clip' : 1.0, 
    }
    optimizer = select_optimizer(
        opt_method, optimizer_args,
        batch_size=1, dataset_size=len(train_dset),
    )

    sigma0 = model.get_sigma()
    phiw0 = model.get_parameters()['phi.w']
    confinement_factor0 = model.confinement_factor

    model = train_model(
        model, 
        loss_fn, 
        optimizer,
        train_dataloader, 
        valid_dataloader,
        key=jrandom.PRNGKey(nprng.integers(2**32)),
        num_epochs=1,
        batch_size=1,
        fix_noise=True,
        hyperparams={},
    )

    sigma1 = model.get_sigma()
    phiw1 = model.get_parameters()['phi.w']
    confinement_factor1 = model.confinement_factor

    errors = []
    if not jnp.allclose(sigma0, sigma):
        msg = f"Sigma initialization value does not match sigma0."
        msg += f"\n  Initialization: {sigma}\n  sigma0: {sigma0}"
        errors.append(msg)
    if not jnp.allclose(sigma0, sigma1):
        msg = f"Sigma value before and after training does not match."
        msg += f"\n  Initial: {sigma0}\n  Final: {sigma1}"
        errors.append(msg)
    if not jnp.allclose(confinement_factor0, confinement_factor1):
        msg = f"Confinement factor before and after training does not match."
        msg += f"\n  Initial: {confinement_factor0}\n  Final: {confinement_factor1}"
        errors.append(msg)
    for arr0, arr1 in zip(phiw0, phiw1):
        if jnp.allclose(arr0, arr1):
            msg = f"Phi weights before and after training should differ."
            msg += f"\n  Initial: {arr0}\n  Final: {arr1}"
            errors.append(msg)
    
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    
"""Tests of model training

"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

from plnn.models import DeepPhiPLNN
from plnn.dataset import get_dataloaders
from plnn.loss_functions import mean_cov_loss, kl_divergence_loss
from plnn.optimizers import get_optimizer_args, select_optimizer
from plnn.model_training import train_model
from plnn.dataset import LandscapeSimulationDataset, NumpyLoader


#####################
##  Configuration  ##
#####################

OUTDIR = f"{TMPDIR}/test_training"

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
        dt=0.5, 
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


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('datdir_train, datdir_valid', [
    [f"{DATDIR}/simtest1/data_train", f"{DATDIR}/simtest1/data_valid"],
])
@pytest.mark.parametrize('sigma', [0.3])
@pytest.mark.parametrize('loss_fn', [kl_divergence_loss, mean_cov_loss])
@pytest.mark.parametrize('opt_method', ['sgd', 'adam', 'rms'])
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
        outdir=OUTDIR
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
    
    remove_dir(OUTDIR)

    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    

@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('sample_cells', [False, True])
@pytest.mark.parametrize('fix_noise', [False, True])
@pytest.mark.parametrize('nan_max_attempts, expect_context', [
    [0, pytest.raises(RuntimeError)],
    [1, pytest.raises(RuntimeError)],
    [2, does_not_raise()],
    [3, does_not_raise()],
])
def test_divergent_training(dtype, sample_cells, fix_noise, nan_max_attempts, expect_context):
    """Test a case in which cells diverge.
    
    Initizialization and activation function are such that the phi function is
            phi(x, y) = x^4 + y^4 +x^2 + y^2

    During training, this model should fail because the timestep is too large.
    When that happens, a nan value in the x position will propogate through to
    the loss, and then to the model parameters via the update. At this point,
    the erroring model should be saved to the debugging directory. Importantly,
    this erroring model should have valid parameters, namely the original ones:
        phi.w: [[[1,0],[0,1]], [1, 1]]
        phi.b: [[], []]
        dt0: 1.0
    
    """
    key = jrandom.PRNGKey(0)
    model, hyperparams = DeepPhiPLNN.make_model(
        key,
        dtype=dtype,
        include_phi_bias=False,
        phi_hidden_dims=[2],
        phi_hidden_acts='square',
        phi_final_act=None,
        ndims=2,
        nparams=2,
        nsigs=2,
        ncells=1,
        signal_type='jump',
        nsigparams=3,
        sigma_init=0.0,
        solver='euler',
        dt0=1.0,
        confine=True,
        confinement_factor=1.0,
        include_tilt_bias=False,
        sample_cells=sample_cells,
    )
    key, subkey = jrandom.split(key, 2)
    model = model.initialize(
        subkey, dtype=dtype,
        init_phi_weights_method='explicit',
        init_phi_weights_args=[[[[1, 0],[0, 1]], [[1, 1]]]],
    )

    # Fails when dt=1.0 and dt=0.1, should work when dt=0.01
    training_data = [
        [
            # signal parameters (null)
            [[1, 0, 0],[1, 0, 0]],
            # Datapoints
            [{'t0': 0.0, 
              'x0': [[3,0],[0,0]], 
              't1': 10.0, 
              'x1': [[1., 1.], [1., 1.]]},
              {'t0': 0.0, 
              'x0': [[3,0],[0,0]], 
              't1': 10.0, 
              'x1': [[1., 1.], [1., 1.]]},
            ]
        ]
    ]

    train_dset = LandscapeSimulationDataset(
        data=training_data,
        nsims=1,
    )

    train_dataloader = NumpyLoader(
        train_dset, 
        batch_size=1, 
        shuffle=False,
    )

    valid_dataloader = NumpyLoader(
        train_dset, 
        batch_size=1, 
        shuffle=False,
    )

    optimizer_args = {
        'lr_schedule' : 'constant',
        'learning_rate' : 1,
        'momentum' : 0.9,
        'weight_decay' : 0.5, 
        'clip' : 1.0, 
    }
    optimizer = select_optimizer(
        'rms', optimizer_args,
        batch_size=1, dataset_size=len(train_dset),
    )

    errors = []
    with expect_context:
        model = train_model(
            model, 
            mean_cov_loss, 
            optimizer,
            train_dataloader, 
            valid_dataloader,
            key=jrandom.PRNGKey(0),
            num_epochs=1,
            batch_size=1,
            fix_noise=fix_noise,
            hyperparams=hyperparams,
            outdir=OUTDIR,
            reduce_dt_on_nan=True,
            dt_reduction_factor=0.1,
            reduce_cf_on_nan=False,
            cf_reduction_factor=None,
            nan_max_attempts=nan_max_attempts,
        )

        # Check that model does not contain nan in phi.w
        phiw0 = model.get_parameters()['phi.w'][0]
        phiw1 = model.get_parameters()['phi.w'][1]
        if jnp.any(jnp.isnan(phiw0)):
            msg = "nan encountered in phiw0 of output model."
            errors.append(msg)
        if jnp.any(jnp.isnan(phiw1)):
            msg = "nan encountered in phiw1 of output model."
            errors.append(msg)

        # Check that final dt0 value is correct
        dt0_final_exp = 0.01
        dt0_final = model.dt0
        if not jnp.allclose(dt0_final, dt0_final_exp):
            msg = f"Incorrect dt0 final. "
            msg += f"Expected {dt0_final_exp}. Got {dt0_final}."
            errors.append(msg)

        def check_model(errors, model, dt0_exp=None, 
                        phiw0_exp=None, phiw1_exp=None, s=""):
            if dt0_exp is not None:
                dt0 = model.dt0
                if not jnp.allclose(dt0_final, dt0_final_exp):
                    msg = f"[{s}] Incorrect dt0. Expected {dt0_exp}. Got {dt0}."
                    errors.append(msg)
            if phiw0_exp is not None:
                phiw0 = model.get_parameters()['phi.w'][0]
                if not jnp.allclose(phiw0, phiw0_exp, equal_nan=True):
                    msg = f"[{s}] Incorrect phi.w[0]."
                    msg += f"\nExpected:\n{phiw0_exp}\nGot:\n{phiw0}"
                    errors.append(msg)
            if phiw1_exp is not None:
                phiw1 = model.get_parameters()['phi.w'][1]
                if not jnp.allclose(phiw1, phiw1_exp, equal_nan=True):
                    msg = f"[{s}] Incorrect phi.w[1]."
                    msg += f"\nExpected:\n{phiw1_exp}\nGot:\n{phiw1}"
                    errors.append(msg)
            return errors
                
        m, _ = DeepPhiPLNN.load(
            f"{OUTDIR}/debug/model_1_0_err_prestep0.pth", dtype=dtype)
        errors = check_model(
            errors, m, 
            dt0_exp=1.0, 
            phiw0_exp=jnp.array([[1, 0],[0, 1]], dtype=dtype),
            phiw1_exp=jnp.array([[1, 1]], dtype=dtype),
            s="prestep0"
        )

        m, _ = DeepPhiPLNN.load(
            f"{OUTDIR}/debug/model_1_0_err_poststep0.pth", dtype=dtype)
        errors = check_model(
            errors, m, 
            dt0_exp=1.0, 
            phiw0_exp=jnp.nan * jnp.ones([2, 2], dtype=dtype),
            phiw1_exp=jnp.nan * jnp.ones([1, 2], dtype=dtype),
            s="poststep0"
        )

        m, _ = DeepPhiPLNN.load(
            f"{OUTDIR}/debug/model_1_0_postop0.pth", dtype=dtype)
        errors = check_model(
            errors, m, 
            dt0_exp=0.1, 
            phiw0_exp=jnp.array([[1, 0],[0, 1]], dtype=dtype),
            phiw1_exp=jnp.array([[1, 1]], dtype=dtype),
            s="postop0"
        )

        m, _ = DeepPhiPLNN.load(
            f"{OUTDIR}/debug/model_1_0_err_prestep1.pth", dtype=dtype)
        errors = check_model(
            errors, m, 
            dt0_exp=0.1, 
            phiw0_exp=jnp.array([[1, 0],[0, 1]], dtype=dtype),
            phiw1_exp=jnp.array([[1, 1]], dtype=dtype),
            s="prestep1"
        )

        m, _ = DeepPhiPLNN.load(
            f"{OUTDIR}/debug/model_1_0_err_poststep1.pth", dtype=dtype)
        errors = check_model(
            errors, m, 
            dt0_exp=0.1, 
            phiw0_exp=jnp.nan * jnp.ones([2, 2], dtype=dtype),
            phiw1_exp=jnp.nan * jnp.ones([1, 2], dtype=dtype),
            s="poststep1"
        )

        m, _ = DeepPhiPLNN.load(
            f"{OUTDIR}/debug/model_1_0_postop1.pth", dtype=dtype)
        errors = check_model(
            errors, m, 
            dt0_exp=0.01, 
            phiw0_exp=jnp.array([[1, 0],[0, 1]], dtype=dtype),
            phiw1_exp=jnp.array([[1, 1]], dtype=dtype),
            s="postop1"
        )
        
    remove_dir(OUTDIR)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

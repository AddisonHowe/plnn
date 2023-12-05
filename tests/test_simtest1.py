import pytest
import os, glob
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from plnn.models import PLNN, initialize_model
from plnn.model_training import train_model
from plnn.dataset import get_dataloaders
from plnn.helpers import mean_diff_loss

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


TRAINDIR = "tests/simtest1/data_train"
VALIDDIR = "tests/simtest1/data_valid"
NSIMS_TRAIN = 4
NSIMS_VALID = 4

OUTDIR = "tests/simtest1/tmp_out"

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('dtype, atol', [
    [jnp.float32, 1e-4], 
    [jnp.float64, 1e-6], 
])
@pytest.mark.parametrize('batch_size, batch_sims', [
    [4, [['sim0', 'sim1', 'sim2', 'sim3']]],  # train over all 4 in one batch
])
class TestTwoStepSimulation:

    def _load_data(self, dtype, batch_size):
        return get_dataloaders(
            datdir_train=TRAINDIR,
            datdir_valid=VALIDDIR,
            nsims_train=NSIMS_TRAIN,
            nsims_valid=NSIMS_VALID,
            batch_size_train=batch_size,
            batch_size_valid=batch_size,
            shuffle_train=False,
            shuffle_valid=False,
            ndims=2,
            dtype=dtype,
            return_datasets=True,
        )

    def _get_model(self, dtype):
        # Construct the model
        model = PLNN(
            ndim=2, 
            nsig=2, 
            ncells=4, 
            sigma_init=0,
            dt0=0.1,
            include_phi_bias=False, 
            include_signal_bias=False, 
            hidden_acts='tanh',
            final_act=None,
            hidden_dims=[3,3],
            key=jrandom.PRNGKey(0),
            sample_cells=False,
        )
        model = initialize_model(
            jrandom.PRNGKey(0),
            model, dtype=dtype,
            init_phi_weights_method='explicit',
            init_phi_weights_args=[[W1,W2,W3]],
            init_phi_bias_method='none',
            init_phi_bias_args=[],
            init_tilt_weights_method='explicit',
            init_tilt_weights_args=[[WT1]],
            init_tilt_bias_method='none',
            init_tilt_bias_args=[],
        )
        return model
    
    def _remove_files(self, outdir, name):
        # Remove generated files
        for filename in glob.glob(f"{outdir}/{name}*"):
            os.remove(filename) 
    
    def test_1_epoch_train_full_batch(self, dtype, atol, 
                                      batch_size, batch_sims):
        learning_rate = 0.1
        loss_fn = jax.jit(mean_diff_loss)
        errors1 = []

        train_dloader, valid_dloader, _, _ = self._load_data(dtype, batch_size)
        
        model = self._get_model(dtype)
        optimizer = optax.sgd(
            learning_rate=learning_rate, 
        )

        oldparams = model.get_parameters()

        model = train_model(
            model,
            loss_fn, 
            optimizer,
            train_dloader, 
            valid_dloader,
            key=jrandom.PRNGKey(0),
            num_epochs=1,
            batch_size=batch_size,
            outdir=OUTDIR,
            model_name='tmp_model',
        )

        self._remove_files(OUTDIR, 'tmp_model')

        newparams = model.get_parameters()

        if len(newparams) != 4:
            msg = "Bad length for parameters after training 1 epoch. " + \
                f"Expected 4. Got {len(newparams)}."
            errors1.append(msg)

        # Compute the expected new parameter values by hand
        batch_dl_dw1 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw1.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dw2 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw2.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dw3 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw3.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dwt = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dwt.npy') 
                                for s in b] for b in batch_sims])
        
        batch_dl_dw1_avg = np.mean(batch_dl_dw1, axis=1)
        batch_dl_dw2_avg = np.mean(batch_dl_dw2, axis=1)
        batch_dl_dw3_avg = np.mean(batch_dl_dw3, axis=1)
        batch_dl_dwt_avg = np.mean(batch_dl_dwt, axis=1)

        batch_grad_avgs = [batch_dl_dw1_avg, batch_dl_dw2_avg, 
                           batch_dl_dw3_avg, batch_dl_dwt_avg]
        
        params_after_1_epoch = []  # store parameters after 1 epoch training
        for i in range(4):
            oldparam_act = oldparams[i]
            newparam_act = newparams[i]
            newparam_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
            params_after_1_epoch.append(newparam_exp)
            if not np.allclose(newparam_exp, newparam_act, atol=atol):
                msg = f"Error in w{i}:\nExpected:\n{newparam_exp}\nGot:\n{newparam_act}"
                errors1.append(msg)

        assert not errors1, \
            "Errors occurred in epoch 1:\n{}".format("\n".join(errors1))
        

    def test_2_epoch_train_full_batch(self, dtype, atol, 
                                      batch_size, batch_sims):
        learning_rate = 0.1
        loss_fn = mean_diff_loss
        errors1 = []
        errors2 = []
        
        train_dloader, valid_dloader, _, _ = self._load_data(dtype, batch_size)
        
        model = self._get_model(dtype)
        optimizer = optax.sgd(
            learning_rate=learning_rate, 
        )

        oldparams = model.get_parameters()

        model = train_model(
            model,
            loss_fn, 
            optimizer,
            train_dloader, 
            valid_dloader,
            key=jrandom.PRNGKey(0),
            num_epochs=1,
            batch_size=batch_size,
            outdir=OUTDIR,
            model_name='tmp_model',
        )

        self._remove_files(OUTDIR, 'tmp_model')
        
        newparams = model.get_parameters()

        if len(newparams) != 4:
            msg = "Bad length for parameters after training 1 epoch. " + \
                f"Expected 4. Got {len(newparams)}."
            errors1.append(msg)

        # Compute the expected new parameter values by hand
        batch_dl_dw1 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw1.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dw2 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw2.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dw3 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw3.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dwt = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dwt.npy') 
                                for s in b] for b in batch_sims])
        
        batch_dl_dw1_avg = np.mean(batch_dl_dw1, axis=1)
        batch_dl_dw2_avg = np.mean(batch_dl_dw2, axis=1)
        batch_dl_dw3_avg = np.mean(batch_dl_dw3, axis=1)
        batch_dl_dwt_avg = np.mean(batch_dl_dwt, axis=1)

        batch_grad_avgs = [batch_dl_dw1_avg, batch_dl_dw2_avg, 
                           batch_dl_dw3_avg, batch_dl_dwt_avg]
        
        params_after_1_epoch = []  # store parameters after 1 epoch training
        for i in range(4):
            oldparam_act = oldparams[i]
            newparam_act = newparams[i]
            newparam_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
            params_after_1_epoch.append(newparam_exp)
            if not np.allclose(newparam_exp, newparam_act, atol=atol):
                msg = f"Error in w{i}:\nExpected:\n{newparam_exp}\nGot:\n{newparam_act}"
                errors1.append(msg)
        
        # Create new model to train for 2 epochs
        model2 = self._get_model(dtype)
        optimizer = optax.sgd(
            learning_rate=learning_rate, 
        )

        model2 = train_model(
            model2,
            loss_fn, 
            optimizer,
            train_dloader, 
            valid_dloader,
            key=jrandom.PRNGKey(0),
            num_epochs=2,
            batch_size=batch_size,
            outdir=OUTDIR,
            model_name='tmp_model',
        )

        self._remove_files(OUTDIR, 'tmp_model')

        newparams = model2.get_parameters()
        
        if len(newparams) != 4:
            msg = "Bad length for parameters after training 2 epochs. " + \
                f"Expected 4. Got {len(newparams)}."
            errors2.append(msg)

        # Compute the expected new parameter values by hand
        batch_dl_dw1 = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dw1.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dw2 = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dw2.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dw3 = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dw3.npy') 
                                for s in b] for b in batch_sims])
        batch_dl_dwt = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dwt.npy') 
                                for s in b] for b in batch_sims])
        
        batch_dl_dw1_avg = np.mean(batch_dl_dw1, axis=1)
        batch_dl_dw2_avg = np.mean(batch_dl_dw2, axis=1)
        batch_dl_dw3_avg = np.mean(batch_dl_dw3, axis=1)
        batch_dl_dwt_avg = np.mean(batch_dl_dwt, axis=1)

        batch_grad_avgs = [batch_dl_dw1_avg, batch_dl_dw2_avg, 
                        batch_dl_dw3_avg, batch_dl_dwt_avg]
        
        for i in range(4):
            oldparam_act = params_after_1_epoch[i]
            newparam_act = newparams[i]
            new_param_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
            if not np.allclose(new_param_exp, newparam_act, atol=atol):
                msg = f"Error in w{i}:\nExpected:\n{new_param_exp}\nGot:\n{newparam_act}"
                errors2.append(msg)

        assert not errors1, \
            "Errors occurred in epoch 1:\n{}".format("\n".join(errors1))
        assert not errors2, \
            "Errors occurred in epoch 2:\n{}".format("\n".join(errors2))
        
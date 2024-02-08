import pytest
import os, glob
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import optax
from plnn.models import DeepPhiPLNN
from plnn.model_training import train_model
from plnn.dataset import get_dataloaders
from plnn.loss_functions import mean_diff_loss

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


TRAINDIR = "tests/simtest2/data_train"
VALIDDIR = "tests/simtest2/data_valid"
NSIMS_TRAIN = 4
NSIMS_VALID = 4

OUTDIR = "tests/simtest2/tmp_out"

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('dtype, atol', [
    [jnp.float32, 1e-4], 
    [jnp.float64, 1e-6], 
])
@pytest.mark.parametrize('batch_size, batch_sims, final_ws_exp', [
    [2, [['sim0', 'sim1'], ['sim2', 'sim3']],  # train over 2 batches
     [# W1 final expected
        [[1.02157,2.991],  
         [1.9923,1.9557],
         [3.02391,0.994084]],
      # W2 final expected
        [[0.876646,0.892136,-2.11692],
         [-0.102348,0.913638,-0.0664326],
         [-1.01595,1.98658,0.986411]],
      # W3 final expected
        [[2.00485,2.96534,0.959294]],
      # WT final expected
        [[2.06403,3.95404],[-0.991957,0.962239]]]
    ],  
])
class TestTraining:

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
        model = DeepPhiPLNN(
            ndims=2, 
            nparams=2, 
            nsigs=2, 
            signal_type='jump', 
            nsigparams=3, 
            ncells=4, 
            sigma_init=0,
            solver='euler',
            dt0=0.1,
            confine=False,
            sample_cells=False,
            infer_metric=False,
            include_phi_bias=False, 
            include_tilt_bias=False, 
            include_metric_bias=False, 
            phi_hidden_dims=[3,3],
            phi_hidden_acts='tanh',
            phi_final_act=None,
            phi_layer_normalize=False,
            tilt_hidden_dims=[],
            tilt_hidden_acts=None,
            tilt_final_act=None,
            tilt_layer_normalize=False,
            metric_hidden_dims=[],
            metric_hidden_acts=None,
            metric_final_act=None,
            metric_layer_normalize=False,
            key=jrandom.PRNGKey(0),
        )
        model = model.initialize(
            jrandom.PRNGKey(0),
            dtype=dtype,
            init_phi_weights_method='explicit',
            init_phi_weights_args=[[W1,W2,W3]],
            init_phi_bias_method='none',
            init_phi_bias_args=[],
            init_tilt_weights_method='explicit',
            init_tilt_weights_args=[[WT1]],
            init_tilt_bias_method='none',
            init_tilt_bias_args=[],
            init_metric_weights_method='constant',
            init_metric_weights_args=[0.],
            init_metric_bias_method='none',
            init_metric_bias_args=[],
        )
        return model
    
    def _remove_files(self, outdir, name):
        # Remove generated files
        for filename in glob.glob(f"{outdir}/{name}*"):
            os.remove(filename) 
    
    def test_1_epoch_train_2_batch(self, dtype, atol,
                                      batch_size, batch_sims, final_ws_exp):
        learning_rate = 0.1
        loss_fn = mean_diff_loss
        errors1 = []

        train_dloader, valid_dloader, _, _ = self._load_data(dtype, batch_size)
        
        model = self._get_model(dtype)
        optimizer = optax.sgd(
            learning_rate=learning_rate, 
        )

        oldparams = model.get_linear_layer_parameters()

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

        self._remove_files(f"{OUTDIR}/states", 'tmp_model')

        newparams = model.get_linear_layer_parameters()

        if len(newparams) != 4:
            msg = "Bad length for parameters after training 1 epoch. " + \
                f"Expected 4. Got {len(newparams)}."
            errors1.append(msg)
        
        for i in range(4):
            oldparam_act = oldparams[i]
            newparam_act = newparams[i]
            newparam_exp = final_ws_exp[i]
            if not np.allclose(newparam_exp, newparam_act, atol=atol):
                msg = f"Error in w{i}:\nExpected:\n{newparam_exp}\nGot:\n{newparam_act}"
                errors1.append(msg)

        assert not errors1, \
            "Errors occurred in epoch 1:\n{}".format("\n".join(errors1))
        
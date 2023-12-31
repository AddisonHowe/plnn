import pytest
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import AbstractPath
from plnn.models import PLNN, initialize_model
from plnn.helpers import mean_cov_loss, mean_diff_loss

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

def get_model(ws, wts, dtype, sigma=0, seed=0, ncells=4, dt=0.1, 
              sample_cells=False):
    # Construct the model
    model = PLNN(
        ndim=2, 
        nsig=2, 
        ncells=ncells, 
        sigma_init=sigma,
        dt0=dt,
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
        metric_hidden_dims=[3,3],
        metric_hidden_acts='tanh',
        metric_final_act=None,
        metric_layer_normalize=False,
        key=jrandom.PRNGKey(seed),
        sample_cells=sample_cells,
        infer_metric=True,
    )
    model = initialize_model(
        jrandom.PRNGKey(seed+1),
        model, dtype=dtype,
        init_phi_weights_method='explicit',
        init_phi_weights_args=[ws],
        init_phi_bias_method='none',
        init_phi_bias_args=[],
        init_tilt_weights_method='explicit',
        init_tilt_weights_args=[wts],
        init_tilt_bias_method='none',
        init_tilt_bias_args=[],
        init_metric_weights_method='constant',
        init_metric_weights_args=[0.],
        init_metric_bias_method='constant',
        init_metric_bias_args=[0.],
    )
    return model

# class BrownianTestPath(AbstractPath):
    
#     dw: jax.Array

#     def __init__(self, t0, t1, dw):
#         self.dw = dw

#     @property
#     def t0(self):
#         return self.t0

#     @property
#     def t1(self):
#         return self.t1

#     def evaluate(self, t0, t1=None, left=True):
#         return self.dw

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestBatchedCoreLandscapeMethods:

    @pytest.mark.parametrize('ws, wts, sigparams, t, x, f_exp, f_shape_exp', [
        [[W1, W2, W3], [WT1], 
         [[5, 0, 1, 1, -1]], [0],  # 1 batch
         [[[0, 1]]], 
         [[[-0.441403, -0.16003]]], (1, 1, 2)
        ],
        [[W1, W2, W3], [WT1], 
         [[5, 0, 1, 1, -1]], [10],  # 1 batch
         [[[0, 1]]], 
         [[[5.5586, 2.83997]]], (1, 1, 2)
        ],
        [[W1, W2, W3], [WT1], 
         [[5, 0, 1, 1, -1],[5, 0, 1, 1, -1]], [0, 10],  # 2 batches
         [[[0, 1]], [[0, 1]]], 
         [[[-0.441403, -0.16003]], [[5.5586, 2.83997]]], (2, 1, 2)
        ],
        [[W1, W2, W3], [WT1], 
         [[5, 0, 1, 1, -1],[5, 0, 1, 1, -1]], [0, 10],  # 2 batches of 3 cells
         [[[0, 1],[0, 1],[0, 1]], [[0, 1],[0, 1],[0, 1]]], 
         [3*[[-0.441403, -0.16003]], 3*[[5.5586, 2.83997]]], (2, 3, 2)
        ],
    ])
    def test_f(self, dtype, ws, wts, sigparams, t, x, 
               f_exp, f_shape_exp):
        sigparams = jnp.array(sigparams, dtype=dtype)
        t = jnp.array(t, dtype=dtype)
        x = jnp.array(x, dtype=dtype)
        model = get_model(ws, wts, dtype, ncells=x.shape[1])
        f_act = jax.vmap(model.f, 0)(t, x, sigparams)
        assert np.allclose(f_act, f_exp) and f_act.shape == f_shape_exp

    @pytest.mark.parametrize('ws, wts, t, x, sigma, g_exp, g_shape_exp', [
        [[W1, W2, W3], [WT1], [0],  # 1 batch
         [[[0, 1]]], 0.01, 
         [[[0.01, 0.01]]], (1, 1, 2)
        ],
        [[W1, W2, W3], [WT1], [10],  # 1 batch
         [[[0, 1]]], 0.01, 
         [[[0.01, 0.01]]], (1, 1, 2)
        ],
        [[W1, W2, W3], [WT1], [0, 10],  # 2 batches
         [[[0, 1]], [[0, 1]]], 0.01, 
         [[[0.01, 0.01]], [[0.01, 0.01]]], (2, 1, 2)
        ],
        [[W1, W2, W3], [WT1], 
         [0, 10],  # 2 batches of 3 cells
         [[[0, 1],[0, 1],[0, 1]], [[0, 1],[0, 1],[0, 1]]], 0.01, 
         [3*[[0.01, 0.01]], 3*[[0.01, 0.01]]], (2, 3, 2)
        ],
    ])
    @pytest.mark.parametrize('infer_noise', [True, False])
    def test_g(self, dtype, ws, wts, t, x, 
               sigma, infer_noise, g_exp, g_shape_exp):
        model = get_model(ws, wts, dtype, sigma)
        t = jnp.array(t, dtype=dtype)
        x = jnp.array(x, dtype=dtype)
        g_act = jax.vmap(model.g, 0)(t, x)
        assert np.allclose(g_act, g_exp) and g_act.shape == g_shape_exp

    @pytest.mark.parametrize("ws, x, phi_exp, phi_shape_exp", [
        [[W1, W2, W3], [[[0, 0]]], [[0.0]], (1,1)],
        [[W1, W2, W3], [[[0, 1]]], [[3.9934]], (1,1)],
        [[W1, W2, W3], [[[1, 0]]], [[2.69506]], (1,1)],
        [[W1, W2, W3], [[[1, 1]]], [[3.24787]], (1,1)],
        [[W1, W2, W3], [[[0, 0],[0, 1],[1, 0],[1, 1]]], 
         [[0.0, 3.9934, 2.69506, 3.24787]], (1,4)],
        [[W1, W2, W3], 
         [[[0, 0],[0, 1],[1, 0],[1, 1]],
          [[1, 1],[0, 1],[1, 0],[0, 0]]], 
         [[0.0,3.9934,2.69506,3.24787], 
          [3.24787,3.9934,2.69506,0.0]], (2, 4)],
    ])
    def test_phi(self, dtype, ws, x, phi_exp, phi_shape_exp):
        model = get_model(ws, [WT1], dtype=dtype)
        x = jnp.array(x, dtype=dtype)
        phi_act = jax.vmap(model.phi, 0)(x)
        assert np.allclose(phi_exp, phi_act) and phi_act.shape == phi_shape_exp

    @pytest.mark.parametrize("ws, x, grad_phi_exp, shape_exp", [
        [[W1, W2, W3], [[[0, 0]]], [[[6.0, 14.0]]], (1, 1, 2)],
        [[W1, W2, W3], [[[0, 1]]], [[[-3.5586, -0.83997]]], (1, 1, 2)],
        [[W1, W2, W3], [[[1, 0]]], [[[1.11945, 2.71635]]], (1, 1, 2)],
        [[W1, W2, W3], [[[1, 1]]], [[[-0.00409335, 0.0116181]]], (1, 1, 2)],
        [[W1, W2, W3], 
         [[[1, 1]], [[1, 0]]], 
         [[[-0.00409335, 0.0116181]], [[1.11945, 2.71635]]], (2, 1, 2)],
        [[W1, W2, W3], 
         [[[1, 1], [0, 0]],  # 2 batches, 2 cells/batch
         [[1, 0], [0, 1]]], 
         [[[-0.00409335, 0.0116181], [6.0, 14.0]], 
         [[1.11945, 2.71635], [-3.5586, -0.83997]]], (2, 2, 2)],
        [[W1, W2, W3], 
         [[[1, 1], [0, 0]],  # 3 batches, 2 cells/batch
         [[1, 0], [0, 1]],
         [[1, 0], [0, 1]],], 
         [[[-0.00409335, 0.0116181], [6.0, 14.0]], 
         [[1.11945, 2.71635], [-3.5586, -0.83997]],
         [[1.11945, 2.71635], [-3.5586, -0.83997]]], (3, 2, 2)],
    ])
    def test_grad_phi(self, dtype, ws, x, grad_phi_exp, shape_exp):
        model = get_model(ws, [WT1], dtype)
        x = jnp.array(x, dtype=dtype)
        t = jnp.array(x.shape[0]*[0], dtype=dtype)
        grad_phi_exp = jnp.array(grad_phi_exp, dtype=dtype)
        grad_phi_act = jax.vmap(model.grad_phi, 0)(t, x)

        errors = []
        if not np.allclose(grad_phi_exp, grad_phi_act, atol=1e-5):
            msg = f"Value mismatch between grad phi actual and expected."
            msg += f"\nExpected:\n{grad_phi_exp}\nGot:\n{grad_phi_act}"
            errors.append(msg)
        if not (grad_phi_act.shape == shape_exp):
            msg = f"Shape mismatch between grad phi actual and expected."
            msg += f"Expected {shape_exp}. Got {grad_phi_act.shape}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    @pytest.mark.parametrize("ws, x, grad_phi_exp", [
        [[W1, W2, W3], [[[0, 0]]], [[[6.0, 14.0]]]],
        [[W1, W2, W3], [[[0, 1]]], [[[-3.5586, -0.83997]]]],
        [[W1, W2, W3], [[[1, 0]]], [[[1.11945, 2.71635]]]],
        [[W1, W2, W3], [[[1, 1]]], [[[-0.00409335, 0.0116181]]]],
        [[W1, W2, W3], 
        [[[1, 1]], [[1, 0]]], 
        [[[-0.00409335, 0.0116181]], [[1.11945, 2.71635]]]],
        [[W1, W2, W3], 
        [[[1, 1], [0, 0]],  # 2 batches, 2 cells/batch
        [[1, 0], [0, 1]]], 
        [[[-0.00409335, 0.0116181], [6.0, 14.0]], 
        [[1.11945, 2.71635], [-3.5586, -0.83997]]]],
        [[W1, W2, W3], 
        [[[1, 1], [0, 0]],  # 3 batches, 2 cells/batch
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],], 
        [[[-0.00409335, 0.0116181], [6.0, 14.0]], 
        [[1.11945, 2.71635], [-3.5586, -0.83997]],
        [[1.11945, 2.71635], [-3.5586, -0.83997]]]],
    ])
    @pytest.mark.parametrize("ncells", [1,2,3,4])
    @pytest.mark.parametrize("seed", [1,2,3])
    def test_grad_phi_with_shuffle(self, dtype, 
                                   ws, x, grad_phi_exp, ncells, seed):
        npdtype = np.float32 if dtype == jnp.float32 else np.float64
        grad_phi_exp = np.array(grad_phi_exp, dtype=npdtype)
        model = get_model(ws, [WT1], dtype, seed=seed, ncells=ncells,
                          sample_cells=True)

        x = jnp.array(x, dtype=dtype)
        t = jnp.array(x.shape[0]*[0], dtype=dtype)
        x_samp = model._sample_y0(jrandom.PRNGKey(seed), x)
        grad_phi_act = jax.vmap(model.grad_phi, 0)(t, x_samp)

        key = jrandom.PRNGKey(seed)
        nbatches = x.shape[0]
        ncells_input = x.shape[1]
        shape_exp = (nbatches, ncells, x.shape[-1])
        grad_phi_exp_shuffled = np.zeros([nbatches, ncells, x.shape[2]], 
                                         dtype=npdtype)
        
        if ncells_input < ncells:
            for bidx in range(nbatches):
                key, subkey = jrandom.split(key, 2)
                samp_idxs = jrandom.choice(subkey, x.shape[1], (ncells,), True)
                grad_phi_exp_shuffled[bidx,:] = grad_phi_exp[bidx,samp_idxs]
        else:
            for bidx in range(nbatches):
                key, subkey = jrandom.split(key, 2)
                samp_idxs = jrandom.choice(subkey, x.shape[1], (ncells,), False)
                grad_phi_exp_shuffled[bidx,:] = grad_phi_exp[bidx,samp_idxs]

        errors = []
        if not np.allclose(grad_phi_exp_shuffled, grad_phi_act, atol=1e-6):
            msg = f"Value mismatch between grad phi actual and expected."
            msg += f"Expected:\n{grad_phi_exp_shuffled}\nGot:\n{grad_phi_act}"
            errors.append(msg)
        if not (grad_phi_act.shape == shape_exp):
            msg = f"Shape mismatch between grad phi actual and expected."
            msg += f"Expected {shape_exp}. Got {grad_phi_act.shape}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    @pytest.mark.parametrize('wts, sigparams, t, grad_tilt_exp, shape_exp', [
        [[WT1], [[5, 0, 1, 1, -1]], [0], [[4, 1]], (1, 2)],
        [[WT1], [[5, 0, 1, 1, -1]], [10], [[-2, -2]], (1, 2)],
        [[WT1], 
         [[5, 0, 1, 1, -1],  # 3 batches
          [5, 0, 1, 1, -1],
          [5, 0, 1, 1, -1]], 
         [0, 10, 10], 
         [[4, 1], [-2, -2], [-2, -2]], 
         (3, 2)],
    ])
    def test_grad_tilt(self, dtype, wts, sigparams, t, 
                       grad_tilt_exp, shape_exp):
        sigparams = jnp.array(sigparams, dtype=dtype)
        model = get_model([W1,W2,W3], wts, dtype)
        ts = jnp.array(t, dtype=dtype)
        grad_tilt_act = jax.vmap(model.grad_tilt, 0)(ts, sigparams)

        errors = []
        if not np.allclose(grad_tilt_act, grad_tilt_exp):
            msg = f"Value mismatch between grad tilt actual and expected."
            errors.append(msg)
        if not (grad_tilt_act.shape == shape_exp):
            msg = f"Shape mismatch between grad tilt actual and expected."
            msg += f"Expected {shape_exp}. Got {grad_tilt_act.shape}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


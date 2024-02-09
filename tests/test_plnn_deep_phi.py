"""Tests for DeepPhiPLNN methods.

"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from plnn.models import DeepPhiPLNN

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

@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestBatchedCoreLandscapeMethods:

    @pytest.mark.parametrize('ws, wts, sigparams, t, x, f_exp, f_shape_exp', [
        [[W1, W2, W3], [WT1], 
         [[[5, 0, 1],[5, 1, -1]]], [0],  # 1 batch
         [[[0, 1]]], 
         [[[-0.441403, -0.16003]]], (1, 1, 2)
        ],
        [[W1, W2, W3], [WT1], 
         [[[5, 0, 1],[5, 1, -1]]], [10],  # 1 batch
         [[[0, 1]]], 
         [[[5.5586, 2.83997]]], (1, 1, 2)
        ],
        [[W1, W2, W3], [WT1], 
         [[[5, 0, 1],[5, 1, -1]],[[5, 0, 1],[5, 1, -1]]], [0, 10],  # 2 batches
         [[[0, 1]], [[0, 1]]], 
         [[[-0.441403, -0.16003]], [[5.5586, 2.83997]]], (2, 1, 2)
        ],
        [[W1, W2, W3], [WT1], 
         [[[5, 0, 1],[5, 1, -1]],[[5, 0, 1],[5, 1, -1]]], [0, 10],  # 2 batches of 3 cells
         [[[0, 1],[0, 1],[0, 1]], [[0, 1],[0, 1],[0, 1]]], 
         [3*[[-0.441403, -0.16003]], 3*[[5.5586, 2.83997]]], (2, 3, 2)
        ],
    ])
    def test_f(self, dtype, ws, wts, sigparams, t, x, 
               f_exp, f_shape_exp):
        rtol = 1e-5 if dtype == jnp.float64 else 1e-5
        atol = 1e-8 if dtype == jnp.float64 else 1e-8
        sigparams = jnp.array(sigparams, dtype=dtype)
        t = jnp.array(t, dtype=dtype)
        x = jnp.array(x, dtype=dtype)
        model = get_model(ws, wts, dtype, ncells=x.shape[1])
        f_act = jax.vmap(model.f, 0)(t, x, sigparams)
        errors = []
        if not np.allclose(f_act, f_exp, rtol=rtol, atol=atol):
            msg = f"Value mismatch between f actual and expected."
            msg += f"\nExpected:\n{f_exp}\nGot:\n{f_act}"
            errors.append(msg)
        if not f_act.shape == f_shape_exp:
            msg = f"Shape mismatch between f actual and expected."
            msg += f"Expected {f_shape_exp}. Got {f_act.shape}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

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
    def test_g(self, dtype, ws, wts, t, x, 
               sigma, g_exp, g_shape_exp):
        model = get_model(ws, wts, dtype, sigma)
        t = jnp.array(t, dtype=dtype)
        x = jnp.array(x, dtype=dtype)
        g_act = jax.vmap(model.g, 0)(t, x)
        assert np.allclose(g_act, g_exp) and g_act.shape == g_shape_exp

    @pytest.mark.parametrize("ws, x, phi_exp, phi_shape_exp", [
        [[W1, W2, W3], [[[0, 0]]], [[0.0]], (1,1)],
        [[W1, W2, W3], [[[0, 1]]], [[3.99340437124]], (1,1)],
        [[W1, W2, W3], [[[1, 0]]], [[2.69505521004]], (1,1)],
        [[W1, W2, W3], [[[1, 1]]], [[3.2478696918]], (1,1)],
        [[W1, W2, W3], [[[0, 0],[0, 1],[1, 0],[1, 1]]], 
         [[0.0, 3.99340437124, 2.69505521004, 3.2478696918]], (1,4)],
        [[W1, W2, W3], 
         [[[0, 0],[0, 1],[1, 0],[1, 1]],
          [[1, 1],[0, 1],[1, 0],[0, 0]]], 
         [[0.0,3.99340437124,2.69505521004,3.2478696918], 
          [3.2478696918,3.99340437124,2.69505521004,0.0]], (2, 4)],
    ])
    def test_phi(self, dtype, ws, x, phi_exp, phi_shape_exp):
        rtol = 1e-5 if dtype == jnp.float64 else 1e-5
        atol = 1e-8 if dtype == jnp.float64 else 1e-8
        model = get_model(ws, [WT1], dtype=dtype)
        x = jnp.array(x, dtype=dtype)
        phi_act = jax.vmap(model.phi, 0)(x)
        errors = []
        if not np.allclose(phi_act, phi_exp, rtol=rtol, atol=atol):
            msg = f"Value mismatch between phi actual and expected."
            msg += f"\nExpected:\n{phi_exp}\nGot:\n{phi_act}"
            errors.append(msg)
        if not phi_act.shape == phi_shape_exp:
            msg = f"Shape mismatch between f actual and expected."
            msg += f"Expected {phi_shape_exp}. Got {phi_act.shape}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

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
        rtol = 1e-5 if dtype == jnp.float64 else 1e-4
        atol = 1e-8 if dtype == jnp.float64 else 1e-6
        model = get_model(ws, [WT1], dtype)
        x = jnp.array(x, dtype=dtype)
        t = jnp.array(x.shape[0]*[0], dtype=dtype)
        grad_phi_exp = jnp.array(grad_phi_exp, dtype=dtype)
        grad_phi_act = jax.vmap(model.grad_phi, 0)(t, x)

        errors = []
        if not np.allclose(grad_phi_exp, grad_phi_act, rtol=rtol, atol=atol):
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
        [[WT1], [[[5, 0, 1],[5, 1, -1]]], [0], [[4, 1]], (1, 2)],
        [[WT1], [[[5, 0, 1],[5, 1, -1]]], [10], [[-2, -2]], (1, 2)],
        [[WT1], 
         [[[5, 0, 1],[5, 1, -1]],  # 3 batches
          [[5, 0, 1],[5, 1, -1]],
          [[5, 0, 1],[5, 1, -1]]], 
         [0, 10, 10], 
         [[4, 1], [-2, -2], [-2, -2]], 
         (3, 2)],
    ])
    def test_grad_tilt(self, dtype, wts, sigparams, t, 
                       grad_tilt_exp, shape_exp):
        sigparams = jnp.array(sigparams, dtype=dtype)
        model = get_model([W1,W2,W3], wts, dtype, 
                          signal_type='jump', nsigparams=3)
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

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

@pytest.mark.parametrize("dtype, rtol, atol", [
    [jnp.float32, 1e-4, 1e-6],
    [jnp.float64, 1e-5, 1e-8],
])
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
    def test_f(self, dtype, rtol, atol, ws, wts, sigparams, t, x, 
               f_exp, f_shape_exp):
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
    def test_g(self, dtype, rtol, atol, ws, wts, t, x, 
               sigma, g_exp, g_shape_exp):
        model = get_model(ws, wts, dtype, sigma)
        t = jnp.array(t, dtype=dtype)
        x = jnp.array(x, dtype=dtype)
        g_act = jax.vmap(model.g, 0)(t, x)
        assert np.allclose(g_act, g_exp, atol=atol, rtol=rtol) and \
            g_act.shape == g_shape_exp

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
    def test_phi(self, dtype, rtol, atol, ws, x, phi_exp, phi_shape_exp):
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
    def test_grad_phi(self, dtype, rtol, atol, ws, x, grad_phi_exp, shape_exp):
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
    def test_grad_phi_with_shuffle(self, dtype, rtol, atol, 
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
        if not np.allclose(grad_phi_exp_shuffled, grad_phi_act, 
                           rtol=rtol, atol=atol):
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
    def test_grad_tilt(self, dtype, rtol, atol, wts, sigparams, t, 
                       grad_tilt_exp, shape_exp):
        sigparams = jnp.array(sigparams, dtype=dtype)
        model = get_model([W1,W2,W3], wts, dtype, 
                          signal_type='jump', nsigparams=3)
        ts = jnp.array(t, dtype=dtype)
        grad_tilt_act = jax.vmap(model.grad_tilt, 0)(ts, sigparams)

        errors = []
        if not np.allclose(grad_tilt_act, grad_tilt_exp, atol=atol, rtol=rtol):
            msg = f"Value mismatch between grad tilt actual and expected."
            errors.append(msg)
        if not (grad_tilt_act.shape == shape_exp):
            msg = f"Shape mismatch between grad tilt actual and expected."
            msg += f"Expected {shape_exp}. Got {grad_tilt_act.shape}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("ncells_int, ncells_ext, sample, tol, dtype", [
    [100, 100, True, 1e-2, jnp.float64],
    [100, 100, False, 1e-2, jnp.float64],
    [100, 200, True, 1e-2, jnp.float64],
    [100, 200, False, 1e-2, jnp.float64],
    [200, 100, True, 1e-2, jnp.float64],
    [200, 100, False, 1e-2, jnp.float64],
])
@pytest.mark.parametrize("cxx, cxy, cyy", [
    # [1., 0., 1.],
    [1/9, 0, 1/4],
    # [1/4, 1/9, 1/2],
    [1/4, -1/9, 1/2],
])
@pytest.mark.parametrize("sigma, tfin, dt", [
    # [0.01, 100, 0.1],
    [0.10, 100, 0.1],
])
@pytest.mark.parametrize("solver", ['euler', 'heun'])
@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("seed", [1])
def test_steady_state_distribution(
    ncells_int, ncells_ext, sample, tol, dtype,
    cxx, cxy, cyy,
    sigma, tfin, dt,
    solver, batch_size, seed
):
    assert cxx*cyy - cxy**2 > 0, "BAD TEST CONDITION: NOT POSITIVE DEFINITE!"

    c1 = cxx - cxy
    c2 = cyy - cxy
    c3 = cxy

    a = cyy * sigma**2 / (4 * (cxx*cyy - cxy**2))
    b = cxy * sigma**2 / (4 * (cxy**2 - cxx*cyy))
    c = cxx * sigma**2 / (4 * (cxx*cyy - cxy**2))
    cov_exp = jnp.array([[a, b],[b, c]])

    nprng = np.random.default_rng(seed)
    key = jrandom.PRNGKey(nprng.integers(2**32))

    model, _ = DeepPhiPLNN.make_model(
        key,
        dtype=dtype,
        ndims=2,
        nparams=2,
        nsigs=2,
        ncells=ncells_int,
        signal_type='jump',
        nsigparams=3,
        sigma_init=sigma,
        solver=solver,
        dt0=dt,
        confine=False,
        confinement_factor=1,
        sample_cells=sample,
        include_phi_bias=False,
        include_tilt_bias=False,
        phi_hidden_dims=[3,3],
        phi_hidden_acts=[None, 'square'],
        phi_final_act=None,
        phi_layer_normalize=False,
        tilt_hidden_dims=[],
        tilt_hidden_acts=None,
        tilt_final_act=None,
        tilt_layer_normalize=False,
    )

    key, subkey = jrandom.split(key, 2)

    model = model.initialize(
        subkey, dtype=dtype,
        init_phi_weights_method='explicit',
        init_phi_weights_args=[[[[1,0],[0,1],[1,1]],
                                [[1,0,0],[0,1,0],[0,0,1]],
                                [[c1,c2,c3]]]],
    )

    key, subkey = jrandom.split(key, 2)
    x0 = jrandom.normal(subkey, [ncells_ext,2])
    x0 = jnp.array(batch_size*[x0], dtype=dtype)

    key, subkey = jrandom.split(key, 2)

    t0 = jnp.array(batch_size*[0], dtype=dtype)
    t1 = jnp.array(batch_size*[tfin], dtype=dtype)
    sigparams=jnp.array(batch_size*[[[0, 0, 0],[0, 0, 0]]], dtype=dtype)

    res = model(t0, t1, x0, sigparams, subkey)

    errors = []

    if batch_size > 1:
        # Check pairwise difference between batches
        for i in range(batch_size - 1):
            for j in range(i+1, batch_size):
                if jnp.allclose(res[i], res[j]):
                    msg = f"Batch {i} and {j} resulted in the same output."
                    errors.append(msg)

    for bidx in range(batch_size):
        x1 = res[bidx]
        cov_act = jnp.cov(x1.T)
        diff = jnp.abs(cov_act - cov_exp)
        frobnorm = jnp.linalg.norm(diff)
        if frobnorm > tol:
            msg = f"Frobenius norm of cov. diff: {frobnorm} > tol={tol:.5g}."
            msg += "\nIMPORTANT: This is a stochastic test and may fail."
            errors.append(msg)
    
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from plnn.helpers import mean_cov_loss, mean_diff_loss, kl_divergence_est

from plnn.models import PLNN, make_model, initialize_model


def test_make():
    key = jrandom.PRNGKey(0)
    model, hyperparams = make_model(
        key, 2, 2, 100, 
        'jump', 5, 0, 
        [16,32,32,16], 
        'tanh', 
        'none', 
        layer_normalize=False, 
        include_phi_bias=True, 
        include_signal_bias=False, 
        sample_cells=True, 
        dt0=1e-2,
        dtype=jnp.float32,
    )
    

@pytest.mark.parametrize('phi_w_method, phi_w_args', [
    ['constant', [0.]],
    ['normal', [0., 1.]],
    ['xavier_uniform', []],
    ['explicit', [[[1],[2],[3],[4],[5]]]],
])
@pytest.mark.parametrize('phibias, phi_b_method, phi_b_args', [
    [True, 'constant', [0.]],
    [True, 'normal', [0., 1.]],
    # [True, 'xavier_uniform', []],  # error: cannot use xavier for bias
    [False, 'constant', [0.]],
    [False, 'normal', [0., 1.]],
    # [False, 'xavier_uniform', []],
])
@pytest.mark.parametrize('tilt_w_method, tilt_w_args', [
    ['constant', [0.]],
    ['normal', [0., 1.]],
    ['xavier_uniform', []],
])
@pytest.mark.parametrize('sigbias, tilt_b_method, tilt_b_args', [
    [True, 'constant', [0.]],
    [True, 'normal', [0., 1.]],
    # [True, 'xavier_uniform', []],  # error: cannot use xavier for bias
    [False, 'constant', [0.]],
    [False, 'normal', [0., 1.]],
    # [False, 'xavier_uniform', []],
])
def test_init(phi_w_method, phi_w_args, 
              phibias, phi_b_method, phi_b_args, 
              tilt_w_method, tilt_w_args,
              sigbias, tilt_b_method, tilt_b_args):
    key = jrandom.PRNGKey(0)
    model, hyperparams = make_model(
        key, 2, 2, 100, 
        'jump', 5, 0, 
        [16,32,32,16], 'tanh', 'none', 
        layer_normalize=False, 
        include_phi_bias=phibias, 
        include_signal_bias=sigbias,
        sample_cells=True,
        dt0=1e-2,
        dtype=jnp.float32,
    )
    key, subkey = jrandom.split(key, 2)
    new_model = initialize_model(
        subkey, 
        model, 
        init_phi_weights_method=phi_w_method,
        init_phi_weights_args=phi_w_args,
        init_phi_bias_method=phi_b_method,
        init_phi_bias_args=phi_b_args,
        init_tilt_weights_method=tilt_w_method,
        init_tilt_weights_args=tilt_w_args,
        init_tilt_bias_method=tilt_b_method,
        init_tilt_bias_args=tilt_b_args,
    )
    
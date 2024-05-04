"""Loss function tests.

"""

import pytest
import numpy as np
import jax.numpy as jnp
from plnn.loss_functions import mean_cov_loss, mean_diff_loss, kl_divergence_loss

from tests.conftest import DATDIR, TMPDIR

#####################
##  Configuration  ##
#####################

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('y_sim, y_obs, loss_exp', [
    [   
        # Simulated
        [[[0,0],[1,1],[1,0],[0,1]],
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Observed
        [[[1,1],[0,0],[1,0],[0,1]],
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Loss
        (0 + 0 + 0)/3
    ],
    [   
        # Simulated
        [[[0,0],[1,1],[1,0],[0,1]],
         [[1,1],[1,1],[1,1],[1,1]],
         [[0,0],[0,0],[0,0],[0,0]]],
        # Observed
        [[[1,1],[0,0],[1,0],[0,1]],
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Loss
        ((0+0) + (2+0) + (2+0))/3
    ],
    [   
        # Simulated
        [[[0,0],[2,2],[2,0],[0,2]],  # mean: [1,1]  cov: [1.333,0,0,1.333]
         [[1,1],[1,1],[1,1],[1,1]],
         [[0,0],[0,0],[0,0],[0,0]]],
        # Observed
        [[[1,1],[0,0],[1,0],[0,1]],  # mean: [0.5,0.5]  cov: [0.333,0,0,0.333]
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Loss
        ((0.5+2) + (2+0) + (2+0))/3
    ],
])
def test_mean_cov_loss(y_sim, y_obs, loss_exp):
    y_sim = jnp.array(y_sim)
    y_obs = jnp.array(y_obs)
    loss_act = mean_cov_loss(y_sim, y_obs)
    assert np.allclose(loss_act, loss_exp)


@pytest.mark.parametrize('dtype, atol', [
    [jnp.float32, 1e-6], 
    [jnp.float64, 1e-6]
])
@pytest.mark.parametrize('xpath, ypath, loss_exp', [
    ['loss_test_data/xtest1.npy', 
     'loss_test_data/ytest1.npy',
     3.061594942272676],
    ['loss_test_data/ytest1.npy', 
     'loss_test_data/xtest1.npy',
     2.5479272873648875],
    ['loss_test_data/xtest2.npy', 
     'loss_test_data/ytest2.npy',
     0.8700861617889799],
    ['loss_test_data/xtest3.npy', 
     'loss_test_data/ytest3.npy',
     -0.22314355131420965],
])
def test_kl_loss(dtype, atol, xpath, ypath, loss_exp):
    x_sim = jnp.array(np.load(f"{DATDIR}/{xpath}"), dtype=dtype)
    y_obs = jnp.array(np.load(f"{DATDIR}/{ypath}"), dtype=dtype)
    loss_act = kl_divergence_loss(x_sim, y_obs)  # computed KL(sim|obs)
    assert jnp.allclose(loss_exp, loss_act, atol=atol), \
        f"Expected:\n{loss_exp}\nGot:\n{loss_act}"
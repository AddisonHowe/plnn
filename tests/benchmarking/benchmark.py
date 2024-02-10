"""Benchmarking Tests

pytest -s --benchmark tests/benchmarking/benchmark.py

"""

import pytest
import numpy as np
import timeit
import jax
import jax.numpy as jnp
import jax.random as jrandom

from tests.conftest import DATDIR, TMPDIR

from plnn.models import DeepPhiPLNN
from plnn.model_training import train_model
from plnn.main import select_loss_function, select_optimizer
from plnn.dataset import get_dataloaders

#####################
##  Configuration  ##
#####################

BMDIR = f"{DATDIR}/benchmark_data"

MODEL_FPATH = f"{BMDIR}/benchmark_model.pth"
TRAINDIR = f"{BMDIR}/benchmark_data_1a/training"
VALIDDIR = f"{BMDIR}/benchmark_data_1a/validation"

NSIMS_TRAIN = 20
NSIMS_VALID = 5

OUTDIR = f"{TMPDIR}/benchmarks"

##############################################################################
###########################   BEGIN BENCHMARKING   ###########################
##############################################################################

@pytest.mark.benchmark
class TestBenchmarkModel:

    def test_compare_phi_computation(self):
        model, hyperparams = DeepPhiPLNN.load(MODEL_FPATH)
        print("model dt:", model.dt0)

        key = jrandom.PRNGKey(0)

        train_dataloader, _ = get_dataloaders(
            TRAINDIR, VALIDDIR, NSIMS_TRAIN, NSIMS_VALID,
            batch_size_train=100,
            batch_size_valid=100,
            dtype=jnp.float32,
            shuffle_train=False,
            shuffle_valid=False,
            return_datasets=False
        )

        batch_data = next(iter(train_dataloader))
        data_in, y1 = batch_data
        
        t0, y0, t1, sigparams = data_in
        print("t interval:", t1[0] - t0[0])

        niters = 1

        # Warmup
        time0 = timeit.default_timer()
        y_init = model(t0, t1, y0, sigparams, key)
        jax.block_until_ready(y_init)
        time1 = timeit.default_timer()
        total_time_compile = time1 - time0
        
        # Time with JAX
        time0 = timeit.default_timer()
        for i in range(niters - 1):
            y_pred = model(t0, t1, y0, sigparams, key)
        
        y = model(t0, t1, y0, sigparams, key)
        jax.block_until_ready(y)
        time1 = timeit.default_timer()
        total_time_with_jax = time1 - time0

        print(f"Compilation time: {total_time_compile:.8g}")
        print(f"Time with JAX: {total_time_with_jax / niters:.8g}")


    @pytest.mark.parametrize(
            'id, batch_size, ncells, dt_save, dt, solver, confine, sample', [
        ['run0',    50,     100,    10.,   1e-1,   'euler',    True,   True],
        ['run1',    50,     100,    20.,   1e-1,   'euler',    True,   True],
        ['run2',    50,     100,    100.,  1e-1,   'euler',    True,   True],
        ['run3',    100,    100,    10.,   1e-1,   'euler',    True,   True],
        ['run4',    100,    100,    20.,   1e-1,   'euler',    True,   True],
        ['run5',    100,    100,    100.,  1e-1,   'euler',    True,   True],
        ['run6',    50,     100,    10.,   1e-2,   'euler',    True,   True],
        ['run7',    50,     100,    20.,   1e-2,   'euler',    True,   True],
        ['run8',    50,     100,    100.,  1e-2,   'euler',    True,   True],
        ['run9',    100,    100,    10.,   1e-2,   'euler',    True,   True],
        ['run10',   100,    100,    20.,   1e-2,   'euler',    True,   True],
        ['run11',   100,    100,    100.,  1e-2,   'euler',    True,   True],
    ])
    def test_tensorboard_profile(self, id, batch_size, ncells, dt_save, dt, 
                                 solver, confine, sample):

        NSIGS = 2
        SIGNAL_TYPE = 'sigmoid'
        NSIGPARAMS = 4

        key = jrandom.PRNGKey(0)

        model, hyperparams = DeepPhiPLNN.make_model(
            key, dtype=jnp.float32,
            ndims=2, 
            nparams=2,
            nsigs=NSIGS, 
            ncells=100, 
            signal_type=SIGNAL_TYPE, 
            nsigparams=NSIGPARAMS, 
            sigma_init=1e-1, 
            solver=solver, 
            dt0=dt, 
            confine=confine,
            sample_cells=sample, 
            infer_metric=False,
            include_phi_bias=True, 
            include_tilt_bias=False,
            include_metric_bias=True,
            phi_hidden_dims=[16,32,32,16], 
            phi_hidden_acts='softplus', 
            phi_final_act=None, 
            phi_layer_normalize=False, 
            tilt_hidden_dims=[],
            tilt_hidden_acts=None,
            tilt_final_act=None,
            tilt_layer_normalize=False,
            metric_hidden_dims=[8,8,8,8], 
            metric_hidden_acts='softplus', 
            metric_final_act=None, 
            metric_layer_normalize=False, 
        )

        model = model.initialize(
            key,
            dtype=jnp.float32,
            init_phi_weights_method='xavier_uniform',
            init_phi_weights_args=None,
            init_phi_bias_method='constant',
            init_phi_bias_args=0.,
            init_tilt_weights_method='xavier_uniform',
            init_tilt_weights_args=None,
            init_tilt_bias_method='constant',
            init_tilt_bias_args=0.,
            init_metric_weights_method='xavier_uniform',
            init_metric_weights_args=None,
            init_metric_bias_method='constant',
            init_metric_bias_args=0.,
        )

        print("model dt:", model.dt0)        

        t0 = jnp.arange(batch_size, dtype=jnp.float32)
        t1 = dt_save + t0
        y0 = 0.1 * jrandom.normal(key, [batch_size, ncells, 2])
        sigparams = jrandom.uniform(key, [batch_size, NSIGS, NSIGPARAMS])

        
        niters = 1
        # Warmup
        time0 = timeit.default_timer()
        y_init = model(t0, t1, y0, sigparams, key)
        jax.block_until_ready(y_init)
        time1 = timeit.default_timer()
        total_time_compile = time1 - time0
        
        # Time with JAX
        time0 = timeit.default_timer()
        for i in range(niters - 1):
            y_pred = model(t0, t1, y0, sigparams, key)
        
        jax.profiler.start_trace(f"tmpbm/tensorboard/{id}")
        y = model(t0, t1, y0, sigparams, key)
        jax.block_until_ready(y)
        jax.profiler.stop_trace()
        time1 = timeit.default_timer()
        total_time_with_jax = time1 - time0

        print(f"\nPROFILE RUN: {id}")
        print(f"\tCompilation time: {total_time_compile:.8g}")
        print(f"\tTime with JAX: {total_time_with_jax / niters:.8g}")




    def _broken_tensorboard_run(self):
        model, hyperparams = DeepPhiPLNN.load(MODEL_FPATH)

        
        xarr = [[[0, 0],[0, 1],[1, 0],[1, 1]],
             [[1, 1],[0, 1],[1, 0],[0, 0]]]
        x = jnp.array(xarr, dtype=jnp.float32)
        _ = jax.vmap(model.phi, 0)(x)

        jax.profiler.start_trace("tmpbm/tensorboard")
        key = jrandom.PRNGKey(0)
        x = jrandom.normal(key, (5000, 5000))
        y = x @ x
        
        y = jax.vmap(model.phi, 0)(x)
        y.block_until_ready()
        
        jax.profiler.stop_trace()

        # seed = 12345
        # seed = seed if seed else np.random.randint(2**32)
        # rng = np.random.default_rng(seed=seed)
        # key = jrandom.PRNGKey(int(rng.integers(2**32)))
        # key, trainkey = jrandom.split(key, 2)
        
        # train_dataloader, valid_dataloader = get_dataloaders(
        #     TRAINDIR, VALIDDIR, NSIMS_TRAIN, NSIMS_VALID,
        #     batch_size_train=10,
        #     batch_size_valid=10,
        #     dtype=jnp.float32,
        #     shuffle_train=False,
        #     shuffle_valid=False,
        #     return_datasets=False
        # )

        # optimizer_args = {
        #     'learning_rate' : 1e-3,
        #     'momentum'      : 0.9,
        #     'weight_decay'  : 0.,
        #     'lr_schedule'   : 'exponential_decay',
        #     'clip'          : 1.0,
        # }

        # loss_fn = select_loss_function("kl")
        # optimizer = select_optimizer('rms', optimizer_args)
        
        
        # train_model(
        #     model,
        #     loss_fn, 
        #     optimizer,
        #     train_dataloader, 
        #     valid_dataloader,
        #     key=trainkey,
        #     num_epochs=3,
        #     batch_size=10,
        #     model_name="benchmark_model",
        #     hyperparams=hyperparams,
        #     outdir="out/tmp/benchmark",
        #     save_all=False,
        #     plotting=False,
        #     plotting_opts={},  # Default plotting options
        #     report_every=10,
        # )
        # x = jrandom.normal(key, (2, 2))
        # x.block_until_ready()

        
        

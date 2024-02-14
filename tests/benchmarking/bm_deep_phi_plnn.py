"""Benchmarking DeepPhiPLNN

pytest -s --benchmark tests/benchmarking/bm_deep_phi_plnn.py

"""

import pytest
import os
import csv
import numpy as np
import timeit
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from tests.conftest import DATDIR, TMPDIR

from plnn.models import DeepPhiPLNN
from plnn.dataset import get_dataloaders

#####################
##  Configuration  ##
#####################

BMDIR = f"{DATDIR}/benchmark_data"

MODEL_FPATH = f"{BMDIR}/benchmark_models/benchmark_model_1b.pth"

TRAINDIR = f"{BMDIR}/benchmark_data_1b/training"
VALIDDIR = f"{BMDIR}/benchmark_data_1b/validation"

NSIMS_TRAIN = 20
NSIMS_VALID = 5

OUTDIR = f"{TMPDIR}/benchmarks"

os.makedirs(OUTDIR, exist_ok=True)

##############################################################################
###########################   BEGIN BENCHMARKING   ###########################
##############################################################################

@pytest.mark.benchmark
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("ncells", [100, 200, 400])
@pytest.mark.parametrize("sample_cells", [False, True])
@pytest.mark.parametrize("solver", ['euler', 'heun'])
@pytest.mark.parametrize("confine", [True])
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.05, 0.10])
@pytest.mark.parametrize("batch_size", [5, 10, 20, 40])
class TestBenchmarkCoreLandscapeMethods:

    NITERS = 20
    NSIGS = 2
    SIGNAL_TYPE = 'sigmoid'
    NSIGPARAMS = 4
    
    TIME_WITHOUT_JIT = False
    RESULTS_LIST_HEADER = [
        'method', 'dtype', 'ncells', 'sample_cells', 
        'solver', 'confine', 'dt', 'batch_size',
        'time_with_jit', 'std_time_with_jit', 
        'time_without_jit','std_time_without_jit',
    ]
    RESULTS_FPATH = f"{OUTDIR}/results.csv"
    RESULTS_LIST = []

    def _get_model(
            self, *, 
            dtype, ncells, solver, dt, confine, sample_cells,
            seed=None,
    ):
        nprng = np.random.default_rng(seed=seed)
        key = jrandom.PRNGKey(nprng.integers(2**32))

        model, _ = DeepPhiPLNN.make_model(
            key, dtype=dtype,
            ndims=2, 
            nparams=2,
            nsigs=self.NSIGS, 
            ncells=ncells, 
            signal_type=self.SIGNAL_TYPE, 
            nsigparams=self.NSIGPARAMS, 
            sigma_init=1e-1, 
            solver=solver, 
            dt0=dt, 
            confine=confine,
            sample_cells=sample_cells, 
            include_phi_bias=True, 
            phi_hidden_dims=[16,32,32,16], 
            phi_hidden_acts='softplus', 
            phi_final_act=None, 
            phi_layer_normalize=False, 
            include_tilt_bias=False,
            tilt_hidden_dims=[],
            tilt_hidden_acts=None,
            tilt_final_act=None,
            tilt_layer_normalize=False,
        )

        model = model.initialize(
            key,
            dtype=dtype,
            init_phi_weights_method='xavier_uniform',
            init_phi_weights_args=None,
            init_phi_bias_method='constant',
            init_phi_bias_args=0.,
            init_tilt_weights_method='xavier_uniform',
            init_tilt_weights_args=None,
            init_tilt_bias_method='constant',
            init_tilt_bias_args=0.,
        )
        return model

    def _get_dataloader(self, dtype, batch_size):
        train_dataloader, _ = get_dataloaders(
            TRAINDIR, VALIDDIR, NSIMS_TRAIN, NSIMS_VALID,
            batch_size_train=batch_size,
            batch_size_valid=batch_size,
            dtype=dtype,
            shuffle_train=False,
            shuffle_valid=False,
            return_datasets=False
        )
        return train_dataloader

    def _get_method_to_time(self, method_name, model):
        if method_name == "forward":
            def bmfunc(t0, t1, y0, sigparams, key):
                return model(t0, t1, y0, sigparams, key)
        elif method_name == "f":
            def bmfunc(t0, t1, y0, sigparams, key):
                return model.f(t0[0], y0[0], sigparams[0])
        elif method_name == "g":
            def bmfunc(t0, t1, y0, sigparams, key):
                return model.g(t0[0], y0[0])
        elif method_name == "phi":
            def bmfunc(t0, t1, y0, sigparams, key):
                return model.phi(y0[0])
        elif method_name == "grad_phi":
            def bmfunc(t0, t1, y0, sigparams, key):
                return model.grad_phi(t0[0], y0[0])
        elif method_name == "tilted_phi":
            def bmfunc(t0, t1, y0, sigparams, key):
                return model.tilted_phi(t0[0], y0[0], sigparams[0])
        else:
            raise RuntimeError(f"Bad method name {method_name}")
        return bmfunc

    def _print_header(self, method, niters, dtype, ncells, sample_cells, 
                      solver, confine, dt, batch_size):
        print(f"\nBenchmarking {method}:")
        print( "********************************")
        print(f"Averaging execution times over {niters} trials.")
        print(f"dtype       : {dtype}")
        print(f"ncells      : {ncells}")
        print(f"sample_cells: {sample_cells}")
        print(f"solver      : {solver}")
        print(f"confine     : {confine}")
        print(f"dt          : {dt}")
        print(f"batch_size  : {batch_size}")
        print( "--------------------------------")
    
    @pytest.mark.parametrize('method', [
        'f', 'g', 'phi', 'grad_phi', 'tilted_phi', 'forward', 
    ])
    def test_benchmark_forward(
            self, dtype, ncells, sample_cells, solver, confine, dt, batch_size,
            method
    ):

        self._print_header(
            method, self.NITERS, 
            dtype, ncells, sample_cells, solver, confine, dt, batch_size
        )

        dict_key = (
            method, 'float32' if dtype == jnp.float32 else 'float64', 
            ncells, sample_cells, solver, confine, dt, batch_size
        )

        key = jrandom.PRNGKey(0)

        model = self._get_model(
            dtype=dtype, ncells=ncells, solver=solver, dt=dt, 
            confine=confine, sample_cells=sample_cells,
        )

        train_dataloader = self._get_dataloader(dtype, batch_size=batch_size)

        batch_data = next(iter(train_dataloader))
        data_in, y1 = batch_data
        t0, y0, t1, sigparams = data_in

        bmfunc_nojit = self._get_method_to_time(method, model)
        
        @eqx.filter_jit
        def bmfunc(t0, t1, y0, sigparams, key):
            return bmfunc_nojit(t0, t1, y0, sigparams, key)

        # Warmup
        time0 = timeit.default_timer()
        y_init = bmfunc(t0, t1, y0, sigparams, key)
        jax.block_until_ready(y_init)
        time1 = timeit.default_timer()
        total_time_compile = time1 - time0
        
        # Time with jit
        times = []
        for i in range(self.NITERS - 1):
            time0 = timeit.default_timer()
            y = bmfunc(t0, t1, y0, sigparams, key).block_until_ready()
            time1 = timeit.default_timer()
            times.append(time1 - time0)
        avg_time_with_jit = np.mean(times)
        std_time_with_jit = np.std(times)


        # Time without jit
        times = []
        if self.TIME_WITHOUT_JIT:
            with jax.disable_jit():
                for i in range(self.NITERS):
                    time0 = timeit.default_timer()
                    y = bmfunc(t0, t1, y0, sigparams, key).block_until_ready()
                    time1 = timeit.default_timer()
                    times.append(time1 - time0)
                avg_time_wo_jit = np.mean(times)
                std_time_wo_jit = np.std(times)
        else:
            avg_time_wo_jit = np.nan
            std_time_wo_jit = np.nan

        # Print results
        print(f"compile time: {total_time_compile:.6g} sec")
        print(f"avg execution time w/ jit : {avg_time_with_jit:.6g} sec")
        if self.TIME_WITHOUT_JIT:
            print(f"avg execution time w/o jit: {avg_time_wo_jit:.6g} sec")
        print( "********************************\n")

        self.RESULTS_LIST.append([
            *dict_key, 
            avg_time_with_jit, std_time_with_jit, 
            avg_time_wo_jit, std_time_wo_jit,
        ])
        
        with open(self.RESULTS_FPATH, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.RESULTS_LIST_HEADER)
            writer.writerows(self.RESULTS_LIST)

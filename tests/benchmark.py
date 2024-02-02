"""Benchmarking Tests

pytest -s --benchmark tests/benchmark.py

"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from plnn.models import PLNN, load_model, initialize_model
from plnn.model_training import train_model
from plnn.main import select_loss_function, select_optimizer
from plnn.dataset import get_dataloaders

MODEL_FPATH = "data/benchmark_data/benchmark_model.pth"
TRAINDIR = "data/benchmark_data/benchmark_data_train"
VALIDDIR = "data/benchmark_data/benchmark_data_valid"
NSIMS_TRAIN = 20
NSIMS_VALID = 5

OUTDIR = "tests/benchmark/tmp_out"


@pytest.mark.benchmark
class TestBenchmarkModel:

    def test_bm_forward(self):
        model, hyperparams = load_model(MODEL_FPATH)

        seed = seed if seed else np.random.randint(2**32)
        rng = np.random.default_rng(seed=seed)
        key = jrandom.PRNGKey(int(rng.integers(2**32)))
        key, trainkey = jrandom.split(key, 2)
        
        train_dataloader, valid_dataloader = get_dataloaders(
            TRAINDIR, VALIDDIR, NSIMS_TRAIN, NSIMS_VALID,
            batch_size_train=10,
            batch_size_valid=10,
            dtype=jnp.float32,
            shuffle_train=False,
            shuffle_valid=False,
            return_datasets=False
        )

        loss_fn = select_loss_function("kl")
        optimizer = select_optimizer('rms', )
        
        jax.profiler.start_trace("tmpbm/tensorboard")
        train_model(
            model,
            loss_fn, 
            optimizer,
            train_dataloader, 
            valid_dataloader,
            key=trainkey,
            num_epochs=10,
            batch_size=10,
            model_name="benchmark_model",
            hyperparams=hyperparams,
            outdir="out/tmp/benchmark",
            save_all=False,
            plotting=False,
            plotting_opts={},  # Default plotting options
            report_every=10,
        )
        x = jnp.random.normal(key, (2, 2))
        x.block_until_ready()

        jax.profiler.stop_trace()
        

"""Model Evaluation Functions

"""

import argparse
import os, sys, warnings
import numpy as np
import time
import tqdm

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx

from plnn.dataset import get_dataloaders
from plnn.model_training import validation_step
from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.models import DeepPhiPLNN
from plnn.loss_functions import select_loss_function



def get_loader(
        dataset_key, *, 
        datdir_train, datdir_valid, datdir_test, 
        nsims_train, nsims_valid, nsims_test,
        length_multiplier,
        ncells_sample,
        seed=None,
):
    train_loader, valid_loader, test_loader, _, _, _ = get_dataloaders(
        datdir_train, datdir_valid, nsims_train, nsims_valid,
        return_datasets=True,
        include_test_data=True,
        shuffle_train=False,
        shuffle_valid=False,
        shuffle_test=False,
        datdir_test=datdir_test, 
        nsims_test=nsims_test, 
        batch_size_train=1,  # Needs to be 1 otherwise loss is averaged
        batch_size_valid=1,  
        batch_size_test=1,
        ncells_sample=ncells_sample,
        length_multiplier=length_multiplier,
        seed=seed
    )
    return {
        'train': train_loader, 'valid': valid_loader, 'test': test_loader
    }[dataset_key]


def run_model_on_evaluation_data(
        model, *,
        loss_fn,  
        loader,
        n_reps,
        batch_size,
        key,
        rng=None,
        sigparam_shape=(2, 4),
        disable_pbar=False,
):
    """Run a series of simulations.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(loader)
    times = np.nan * np.ones(n)
    conditions = np.nan * np.ones([n, *sigparam_shape])
    losses = np.nan * np.ones([n, n_reps])

    inputs_array = []
    y1_array = []
    
    for i, data in enumerate(loader):
        inputs, y1 = data
        times[i] = inputs[0][0]
        conditions[i,:] = inputs[-1][0]
        inputs_array.append(inputs)
        y1_array.append(y1)
    
    inputs_array = jax.tree_map(lambda *x: jnp.stack(x), *inputs_array)
    y1_array = jnp.array(y1_array)

    @eqx.filter_jit
    def validation_step_ntimes(n_reps, model, inputs, y1, loss_fn, key):
        subkeys = jrandom.split(key, n_reps)
        return jax.vmap(validation_step, (None, None, None, None, 0))(
            model, inputs, y1, loss_fn, subkeys
        )
        
    @eqx.filter_jit
    def step_ntimes_vectorized(
            n_reps, model, inputs_arr, y1_arr, loss_fn, key,
    ):
        subkeys = jrandom.split(key, len(y1_arr))
        return jax.vmap(validation_step_ntimes, (None, None, 0, 0, None, 0))(
            n_reps, model, inputs_arr, y1_arr, loss_fn, subkeys
        )
    
    nbatches = n // batch_size + (n % batch_size != 0)
    
    print("Running evaluations...")
    count = 0
    for batch_idx in tqdm.tqdm(range(nbatches), disable=disable_pbar):
        time0 = time.time()
        key, subkey = jrandom.split(key, 2)
        idx0 = count
        idx1 = min(count + batch_size, n)
        
        partial_inputs_array = [arr[idx0:idx1] for arr in inputs_array]
        partial_y1_array = y1_array[idx0:idx1]

        time00 = time.time()
        results = step_ntimes_vectorized(
            n_reps, model, partial_inputs_array, partial_y1_array, loss_fn, key
        )
        print(f"  steptime: {time.time() - time00:.5f}")
        losses[idx0:idx1] = results
        count += len(results)
        print(f"  time: {time.time() - time0:.5f}")

    return losses, times, conditions


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['train', 'valid', 'test'])

    parser.add_argument('--nresamp', type=int, required=True)
    parser.add_argument('--nreps', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    
    parser.add_argument('--dt0', type=float, default=None)
    parser.add_argument('--ncells_sample', type=int, default=None)

    parser.add_argument("--modeldir", type=str, required=True,
                        help="Name of the model directory, with timestamp.")
    parser.add_argument("--basedir", type=str, 
                        default="data/trained_models/facs",
                        help="Directory with trained model subdirectories.")
    parser.add_argument("--datdirbase", type=str, required=True,
                        help="Directory with training data subdirectories," \
                             "each one with training/validation/testing dirs.")
    parser.add_argument("--datdir", type=str, required=True,
                        help="Name of the data directory within `datdirbase`")
    parser.add_argument("--outdir", type=str, required=True)
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pbar', action="store_true")
    return parser.parse_args(args)


def main(args):
    print(f"Handled args: {args}")
    
    dataset_key = args.dataset

    nresamp = args.nresamp
    nreps = args.nreps
    batch_size = args.batch_size
    
    dt0 = args.dt0
    ncells_sample = args.ncells_sample
    
    modeldir_name_with_timestamp = args.modeldir
    basedir = args.basedir
    datdirbase = args.datdirbase
    datdir = args.datdir
    baseoutdir = args.outdir

    seed = args.seed
    disable_pbar = not args.pbar

    #~~~~~ Handle random number generators ~~~~~#
    rng = np.random.default_rng(seed=seed)
    key = jrandom.PRNGKey(seed=rng.integers(2**32))

    #~~~~~ Directory structure and housekeeping ~~~~~#
    model_name = modeldir_name_with_timestamp[0:-16]
    modeldir = f"{basedir}/{modeldir_name_with_timestamp}"
    outdir = f"{baseoutdir}/{modeldir_name_with_timestamp}"
    datdir = f"{datdirbase}/{datdir}"

    datdir_train = f"{datdir}/training"
    datdir_valid = f"{datdir}/validation"
    datdir_test = f"{datdir}/testing"

    os.makedirs(outdir, exist_ok=True)

    nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
    nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
    try:
        nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)
    except FileNotFoundError as e:
        msg = f"{e} Reverting to validation data instead."
        warnings.warn(msg)
        datdir_test = f"{datdir}/validation"
        nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)

    
    #~~~~~ Load model and loss function ~~~~~#
    model, _, idx, model_name, model_fpath = load_model_from_directory(
        modeldir, 
        subdir="states",
        idx='best',
        model_class=DeepPhiPLNN,
        dtype=jnp.float64,
    )

    # Load the argument dictionary and training run dictionary
    logged_args, _ = load_model_training_metadata(
        modeldir,
        load_all=True
    )
    
    if ncells_sample is None:
        ncells_sample = logged_args['ncells_sample']

    loss_id = logged_args['loss']
    loss_fn = select_loss_function(
        loss_id, 
        kernel=logged_args.get('kernel'),
        bw_range=logged_args.get('bw_range'),
    )

    print(f"Loaded model `{model_name}` at epoch {idx} from {model_fpath}.")

    #~~~~~ Run evaluation simulations ~~~~~#

    dataloader = get_loader(
        dataset_key, 
        datdir_train=datdir_train,
        datdir_valid=datdir_valid,
        datdir_test=datdir_test, 
        nsims_train=nsims_train,
        nsims_valid=nsims_valid,
        nsims_test=nsims_test,
        length_multiplier=nresamp,
        ncells_sample=ncells_sample,
        seed=rng.integers(2**32),
    )

    if dt0 is not None:
        model = eqx.tree_at(lambda m: m.dt0, model, dt0)
    else:
        dt0 = model.get_dt0()

    key, subkey = jrandom.split(key, 2)
    results = run_model_on_evaluation_data(
        model, 
        loss_fn=loss_fn,
        loader=dataloader,
        n_reps=nreps,
        batch_size=batch_size,
        key=subkey,
        rng=rng,
        disable_pbar=disable_pbar,
    )

    losses, times, conditions = results

    #~~~~~ Reshape losses and save results to output directory ~~~~~#
    ndata_per_cond = len(np.unique(times))
    nconds = int(len(losses) // ndata_per_cond // nresamp)
    print(nconds)
    losses = losses.reshape([nresamp, nconds, ndata_per_cond, nreps])
    losses = losses.transpose(1, 2, 0, 3)    

    np.save(f"{outdir}/losses_{dataset_key}_dt_{dt0}.npy", losses)
    np.save(f"{outdir}/times_{dataset_key}_dt_{dt0}.npy", times)
    np.save(f"{outdir}/conditions_{dataset_key}_dt_{dt0}.npy", conditions)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

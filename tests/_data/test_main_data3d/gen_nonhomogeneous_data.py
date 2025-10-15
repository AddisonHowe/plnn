"""Script to populate nonhomogeneous test data.

Creates two subdirectories in tests/_data/test_main_data/
    - training_data_nonhomogeneous (16 simulations)
    - validation_data_nonhomogeneous (4 simulations)

The sizes of the observed data matrices vary across simulations and timepoints.

"""

import os
import numpy as np

SEED = 132425456

OUTDIR = "tests/_data/test_main_data"

OUTDIR_TRAIN = f"{OUTDIR}/training_data_nonhomogeneous"
OUTDIR_VALID = f"{OUTDIR}/validation_data_nonhomogeneous"

os.makedirs(OUTDIR_TRAIN, exist_ok=True)
os.makedirs(OUTDIR_VALID, exist_ok=True)

rng = np.random.default_rng(seed=SEED)

DIM = 2

NCELLS_TRAIN = np.array([
    [ 8,  7,  8,  9],
    [ 6,  9,  8, 10],
    [ 7,  5,  5,  7],
    [ 7,  9, 10,  6],
    [ 7,  5,  7,  6],
    [ 9,  7,  5,  6],
    [ 7,  7,  8, 10],
    [10, 10,  6,  6],
], dtype=int)

NCELLS_VALID = np.array([
    [ 7,  6,  8,  9],
    [ 6,  8,  8, 5],
    [ 6,  5,  6,  7],
    [ 8,  9, 7,  6],
], dtype=int)

ntimepoints = NCELLS_TRAIN.shape[1]
timepoints = np.linspace(0, 1, ntimepoints, endpoint=True)

def sample_sigparams():
    sp = np.zeros([2, 4], dtype=np.float64)
    sp[:,0] = rng.uniform(low=timepoints[0], high=timepoints[-1], size=2)
    sp[:,1] = rng.uniform(low=0, high=1, size=2)
    sp[:,2] = rng.uniform(low=0, high=1, size=2)
    sp[:,3] = rng.uniform(low=10, high=15, size=2)
    return sp


for ncells_matrix, outdir in zip(
    [NCELLS_TRAIN, NCELLS_VALID], 
    [OUTDIR_TRAIN, OUTDIR_VALID]
):
    nsims = len(ncells_matrix)
    np.savetxt(f"{outdir}/nsims.txt", [nsims], fmt='%d')
    for simidx in range(nsims):
        simdir = f"{outdir}/sim{simidx}"
        os.makedirs(simdir, exist_ok=True)
        sigparams = sample_sigparams()
        xs = []
        for tidx in range(len(timepoints)):
            ncells = ncells_matrix[simidx, tidx]
            x = rng.random([ncells, DIM])
            xs.append(x)
        xs = np.array(xs, dtype=object)
        np.save(f"{simdir}/ts.npy", timepoints)
        np.save(f"{simdir}/xs.npy", xs, allow_pickle=True)
        np.save(f"{simdir}/sigparams.npy", sigparams)

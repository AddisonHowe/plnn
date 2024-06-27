"""Generate sythetic datasets for the dataloader tests.

Generate synthetic datasets and save in tests/_data/test_dataloader/
    dataset1: 8 simulations, each with 20 2-dimensional cells at 11 timepoints
        between 0 and 10. Cells are ordered lexicographically by simulations 
        and timepoints, with coords (0, 1), (1, 2), etc. Assume 2 sigmoidal 
        signals between 0 and the index of the simulation.
    dataset2: 8 simulations, with a different number of cells in each sample.

"""

import os
import numpy as np
from tests.conftest import DATDIR as BASEDATDIR


DATDIR = f"{BASEDATDIR}/test_dataloader"


def build_dataset1():
    outdir = f"{DATDIR}/dataset1"
    nsims = 8
    ncells = 20
    ndim = 2
    ts = np.linspace(0, 10, 11, endpoint=True)

    cellidx = 0
    for simidx in range(nsims):
        simdir = f"{outdir}/sim{simidx}"
        os.makedirs(simdir, exist_ok=True)
        # Construct state data
        xs = np.nan * np.ones([len(ts), ncells, ndim], dtype=np.float64)
        for tidx in range(len(ts)):
            idxs = np.arange(cellidx, cellidx + ncells)
            x = np.tile(idxs, [ndim, 1])
            x += np.arange(ndim)[:,None]
            xs[tidx] = x.T
            cellidx += ncells
        # Construct signal parameters
        sigparams = np.zeros([2, 4], dtype=np.float64)
        sigparams[0] = [5, 0, simidx, 10]
        sigparams[0] = [5, 1, simidx, 10]
        # Save data for simulation
        np.save(f"{simdir}/ts.npy", ts)
        np.save(f"{simdir}/xs.npy", xs)
        np.save(f"{simdir}/sigparams.npy", sigparams)


def build_dataset2():
    outdir = f"{DATDIR}/dataset2"
    nsims = 8
    ndim = 2
    ts = np.linspace(0, 10, 11, endpoint=True)

    cellidx = 0
    ncells_list = []
    for simidx in range(nsims):
        simdir = f"{outdir}/sim{simidx}"
        os.makedirs(simdir, exist_ok=True)
        # Construct state data
        xs = []
        for tidx in range(len(ts)):
            ncells = 10 + 10*simidx + tidx
            ncells_list.append(ncells)
            idxs = np.arange(cellidx, cellidx + ncells)
            x = np.tile(idxs, [ndim, 1])
            x += np.arange(ndim)[:,None]
            xs.append(x.T)
            cellidx += ncells
            print(simidx, tidx, x[0,0], x[0,-1])
        xs = np.array(xs, dtype=object)
        # Construct signal parameters
        sigparams = np.zeros([2, 4], dtype=np.float64)
        sigparams[0] = [5, 0, simidx, 10]
        sigparams[0] = [5, 1, simidx, 10]
        # Save data for simulation
        np.save(f"{simdir}/ts.npy", ts)
        np.save(f"{simdir}/xs.npy", xs)
        np.save(f"{simdir}/sigparams.npy", sigparams)
        np.save(f"{outdir}/ncells_list.npy", ncells_list)


if __name__ == "__main__":
    build_dataset1()
    build_dataset2()
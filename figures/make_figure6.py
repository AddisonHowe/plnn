"""Figure 6 Script

Generate plots used in Figure 6 of the accompanying manuscript, detailing the
preprocessing of FACS data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.style.use('figures/styles/fig6.mplstyle')

SEED = 123

rng = np.random.default_rng(seed=SEED)


OUTDIR = "figures/out/fig6_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches

MINMARKERSIZE = 3
MINMARKER = '.'
MINCOLOR = 'r'

##############################################################################
##############################################################################
##  Make Figure 6

FIGSIZE = (6, 4)

nrows = 5
ncols = 3

signals = [f"sig{i}" for i in range(1, ncols+1)]
times = [0.0, 1.0, 2.0, 3.0, 4.0]

BLOBS = {
    'sig1': {
        times[0]: {
            'n_components': 1,
            'means': [
                [0, 0, 0],
            ],
            'covs': [
                [0.25, 0.25, 0.25],
            ]
        },
        times[1]: {
            'n_components': 3,
            'means': [
                [0, 0, 0],
                [-1, 0, 0],
                [ 1, 0, 0],
            ],
            'covs': [
                [0.75, 0.50, 0.50],
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20],
            ]
        },
        times[2]: {
            'n_components': 2,
            'means': [
                [-1.5, 0, -0.2],
                [ 1.5, 0,  0.2],
            ],
            'covs': [
                [0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25],
            ]
        },
        times[3]: {
            'n_components': 2,
            'means': [
                [-1.5, 0, -0.3],
                [ 1.5, 0,  0.3],
            ],
            'covs': [
                [0.25, 0.25, 0.25],
                [0.1, 0.5, 0.1],
            ]
        },
        times[4]: {
            'n_components': 3,
            'means': [
                [-1.5, 0, -0.3],
                [ 1.5, 0,  0.3],
                [],
            ],
            'covs': [
                [0.25, 0.25, 0.25],
                [0.1, 0.5, 0.1],
                [],
            ]
        },
    }
}


fig = plt.figure(figsize=FIGSIZE)
for i in range(1, nrows*ncols + 1):
    ax = fig.add_subplot(5, 3, i, projection='3d')
axes = fig.get_axes()

for ridx in range(nrows):
    for cidx in range(ncols):
        pass

plt.savefig(f"{OUTDIR}/fig6.pdf", bbox_inches='tight')


##############################################################################
##############################################################################

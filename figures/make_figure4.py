"""Figure 4 Script

Generate plots used in Figure 4 of the accompanying manuscript, detailing the
preprocessing of FACS data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.style.use('figures/styles/fig4.mplstyle')

SEED = 123

rng = np.random.default_rng(seed=SEED)


OUTDIR = "figures/out/fig4_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches

MINMARKERSIZE = 3
MINMARKER = '.'
MINCOLOR = 'r'

##############################################################################
##############################################################################
##  Generate toy data

# TODO: Copy stuff from notebook generating the data.


##############################################################################
##############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('figures/styles/fig_standard.mplstyle')

from cont.binary_choice import get_binary_choice_curves
from cont.binary_flip import get_binary_flip_curves


OUTDIR = "figures/out/tr_study_figs"
ALPHA = 0.5
COL0 = 'orange'
COL1 = 'blue'

sf = 1/2.54  # scale factor from [cm] to inches

FIGSIZE = (4*sf, 4*sf)

XLABEL = '$s_1$'
YLABEL = '$s_2$'

INFO = {
    'tr_study1': {
        'phi': 'phi1',
        'xlims': [-2, 2],
        'ylims': [-1, 3],
        's10': [-0.50, 0.50],
        's20': [ 0.75, 1.00],
        's11': [-1.00, 1.00],
        's21': [-0.25, 0.25],
        'outdir': OUTDIR,
    },
    'tr_study2': {
        'phi': 'phi2',
        'xlims': [-2, 2],
        'ylims': [-2, 2],
        's10': [-1.50, -1.00],
        's20': [-0.75,  0.75],
        's11': [-1.50, -1.00],
        's21': [-0.75,  0.75],
        'outdir': OUTDIR,
    },
    # 'data_phi1_1a': {
    #     'phi': 'phi1',
    #     'xlims': [-2, 2],
    #     'ylims': [-1, 3],
    #     's10': [-0.50, 0.50],
    #     's20': [ 0.50, 1.50],
    #     's11': [-1.00, 1.00],
    #     's21': [-0.50, 0.50],
    #     'outdir': "figures/out/fig3a_out",
    # },
    'data_phi2_1a': {
        'phi': 'phi2',
        'xlims': [-2, 2],
        'ylims': [-2, 2],
        's10': [-1.50, -1.00],
        's20': [-0.75,  0.75],
        's11': [-1.50, -1.00],
        's21': [-0.75,  0.75],
        'outdir': "figures/out/fig3b_out",
    },
}

os.makedirs(OUTDIR, exist_ok=True)

for k in INFO:
    d = INFO[k]
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    if d['phi'] == 'phi1':
        bifcurves, bifcolors = get_binary_choice_curves()
    elif d['phi'] == 'phi2':
        bifcurves, bifcolors = get_binary_flip_curves()

    for curve, color in zip(bifcurves, bifcolors):
        ax.plot(
            curve[:,0], curve[:,1], 
            color=color,
            linestyle='--',
            linewidth=2,
        )
    
    # Plot the prior for initial signal values
    x0 = d['s10'][0]
    y0 = d['s20'][0]
    w = d['s10'][1] - d['s10'][0]
    h = d['s20'][1] - d['s20'][0]
    # Create the rectangle patch with transparency (alpha)
    rectangle = patches.Rectangle(
        (x0, y0), w, h, 
        alpha=ALPHA, color=COL0, fill=None, hatch=4*'/',
    )
    p1 = ax.add_patch(rectangle)
    
    # Plot the prior for final signal values
    x0 = d['s11'][0]
    y0 = d['s21'][0]
    w = d['s11'][1] - d['s11'][0]
    h = d['s21'][1] - d['s21'][0]
    # Create the rectangle patch with transparency (alpha)
    rectangle = patches.Rectangle(
        (x0, y0), w, h, 
        alpha=ALPHA, color=COL1, fill=None, hatch=4*'\\',
    )
    p2 = ax.add_patch(rectangle)

    ax.legend(
        [p1, p2], ['Initial', 'Final'], 
        bbox_to_anchor=(1.05, 1), loc='upper left',
        fontsize='small',
    )
    
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)

    ax.set_xlim(d['xlims'])
    ax.set_ylim(d['ylims'])

    ax.set_title("Signal Prior")

    plt.savefig(f"{d['outdir']}/{k}", bbox_inches='tight', transparent=True)

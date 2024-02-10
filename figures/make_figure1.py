"""Figure 1 Script

Generate plots used in Figure 1 of the accompanying manuscript.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from notebooks.plot_landscapes import plot_landscape

OUTDIR = "out/figures/fig1_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y - p1*x + p2*y

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

##############################################################################
##############################################################################
##  Untilted binary choice landscape.

FIGNAME = "phi1_heatmap_untilted"

r = 4       # box radius
res = 400   # resolution
lognormalize = True
clip = None

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_aspect('equal')

plot_landscape(
    func_phi1_star, r=r, res=res, signal=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"Untilted Landscape $\phi_1^*$",
    cbar_title="$\log\phi$" if lognormalize else "$\phi$",
    ax=ax,
    saveas=f"{OUTDIR}/{FIGNAME}.pdf" if SAVEPLOTS else None,
);

##############################################################################
##############################################################################
##  Untilted binary flip landscape.

FIGNAME = "phi2_heatmap_untilted"

r = 4       # box radius
res = 400   # resolution
lognormalize = True
clip = None

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_aspect('equal')

plot_landscape(
    func_phi2_star, r=r, res=res, signal=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"Untilted Landscape $\phi_2^*$",
    cbar_title="$\log\phi$" if lognormalize else "$\phi$",
    ax=ax,
    saveas=f"{OUTDIR}/{FIGNAME}.pdf" if SAVEPLOTS else None,
);

##############################################################################
##############################################################################
##  Tilted binary choice landscape

FIGNAME = "phi1_heatmap_tilted"  # appended with index i

signals = [
    [ 0.00, 1.0],
    [-0.75, 0.0],
    [ 0.75, 0.0],
]
r = 4       # box radius
res = 400   # resolution
lognormalize = True
clip = None

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_aspect('equal')

for i, signal in enumerate(signals):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal')

    title = f"Tilted Landscape $\phi_1(x,y|\\vec{{p}}=" + \
                f"\langle{signal[0]:.2g},{signal[1]:.2g}\\rangle)$"
    plot_landscape(
        func_phi1_star, r=r, res=res, signal=signal, 
        lognormalize=lognormalize,
        clip=clip,
        title=title,
        cbar_title="$\log\phi$",
        ax=ax,
        saveas=f"{OUTDIR}/{FIGNAME}_{i}.pdf" if SAVEPLOTS else None
    );

##############################################################################
##############################################################################
##  Tilted binary flip landscape

FIGNAME = "phi2_heatmap_tilted"  # appended with index i

signals = [
    [-1.0,  0.5],
    [-1.0, -0.5],
]
r = 4       # box radius
res = 400   # resolution
lognormalize = True
clip = None

for i, signal in enumerate(signals):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal')

    title = f"Tilted Landscape $\phi_2(x,y|\\vec{{p}}=" + \
                f"\langle{signal[0]:.2g},{signal[1]:.2g}\\rangle)$"
    plot_landscape(
        func_phi2_star, r=r, res=res, signal=signal, 
        lognormalize=lognormalize,
        clip=clip,
        title=title,
        cbar_title="$\log\phi$",
        ax=ax,
        saveas=f"{OUTDIR}/{FIGNAME}_{i}.pdf" if SAVEPLOTS else None
    );

##############################################################################
##############################################################################

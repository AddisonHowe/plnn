"""Figure 3 Script

Generate plots used in Figure 3 of the accompanying manuscript.
"""

import os
from glob import glob
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.style.use('figures/fig3.mplstyle')

from plnn.models import DeepPhiPLNN
from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.pl import plot_landscape, plot_phi

from cont.binary_choice import get_binary_choice_curves
from cont.binary_flip import get_binary_flip_curves
from cont.plnn_bifurcations import get_plnn_bifurcation_curves 

SEED = 12345

rng = np.random.default_rng(seed=SEED)


OUTDIR = "out/figures/fig3_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y - p1*x + p2*y

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches

##############################################################################
##############################################################################
##  Untilted binary choice landscape.

FIGNAME = "phi1_heatmap"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

plot_landscape(
    func_phi1_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"$\phi_1^*$",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME}" if SAVEPLOTS else None,
    figsize=FIGSIZE,
)

FIGNAME = "phi1_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves, bifcolors = get_binary_choice_curves(rng=rng)
for curve, color in zip(bifcurves, bifcolors):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)

plt.savefig(f"{OUTDIR}/{FIGNAME}")

##############################################################################
##############################################################################
##  Untilted binary flip landscape.

FIGNAME = "phi2_heatmap"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"$\phi_2^*$",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME}" if SAVEPLOTS else None,
    figsize=FIGSIZE,
)

FIGNAME = "phi2_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves, bifcolors = get_binary_flip_curves(rng=rng)
for curve, color in zip(bifcurves, bifcolors):
    if color == 'k':
        ax.plot(curve[:,0], curve[:,1], '-', color=color, linewidth=1)
    else:
        ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

plt.savefig(f"{OUTDIR}/{FIGNAME}")


##############################################################################
##############################################################################
##  Inferred binary choice.

MODELDIR = "data/trained_models/model_phi1_1a_v1_20240311_130654"

FIGNAME = "phi1_inferred"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

model, hps, idx, name, fpath = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR)

plot_phi(
    model, tilt=[0., 0.], 
    r=r, res=res,
    lognormalize=lognormalize,
    clip=clip,
    title=f"Inferred $\phi_1$",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME}" if SAVEPLOTS else None,
    figsize=FIGSIZE,
)

FIGNAME = "inferred_phi1_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves, bifcolors = get_plnn_bifurcation_curves(
    model, num_starts=100, rng=rng
)
for curve, color in zip(bifcurves, bifcolors):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)

plt.savefig(f"{OUTDIR}/{FIGNAME}")


##############################################################################
##############################################################################
##  Inferred binary flip.

MODELDIR = "data/trained_models/model_phi2_1a_v1_20240311_131333"

FIGNAME = "phi2_inferred"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

model, hps, idx, name, fpath = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR)

plot_phi(
    model, tilt=[0., 0.], 
    r=r, res=res,
    lognormalize=lognormalize,
    clip=clip,
    title=f"Inferred $\phi_2$",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME}" if SAVEPLOTS else None,
    figsize=FIGSIZE,
)

FIGNAME = "inferred_phi2_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves, bifcolors = get_plnn_bifurcation_curves(
    model, 
    num_starts=50, 
    p1lims=[-10, 10],
    p2lims=[-10, 10],
    rng=rng
)
for curve, color in zip(bifcurves, bifcolors):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

plt.savefig(f"{OUTDIR}/{FIGNAME}")


##############################################################################
##############################################################################

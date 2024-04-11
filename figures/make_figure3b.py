"""Figure 3b Script

Generate plots corresponding to those used in Figure 3 of the accompanying 
manuscript, but using the binary flip landscape instead of the binary choice.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.style.use('figures/fig3.mplstyle')

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.pl import plot_landscape, plot_phi

from cont.binary_flip import get_binary_flip_curves
from cont.plnn_bifurcations import get_plnn_bifurcation_curves 

SEED = 1234512345

rng = np.random.default_rng(seed=SEED)


OUTDIR = "figures/out/fig3b_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches


##############################################################################
##############################################################################
##  Binary Flip Landscape

#################################  Heatmap of true landscape
FIGNAME = "phi2_heatmap"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=None,
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME}" if SAVEPLOTS else None,
    figsize=FIGSIZE,
)


#################################  Bifurcation diagram of true landscape
FIGNAME = "phi2_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves, bifcolors = get_binary_flip_curves(rng=rng)
for curve, color in zip(bifcurves, bifcolors):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Load the inferred landscape

MODELDIR = "data/trained_models/model_phi2_1a_v1_20240311_131333"

model, hps, idx, name, fpath = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR)


#################################  Heatmap of inferred landscape

FIGNAME = "phi2_inferred"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None
ax = plot_phi(
    model, tilt=[0., 0.], 
    r=r, res=res,
    lognormalize=lognormalize,
    clip=clip,
    title=None,
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME}" if SAVEPLOTS else None,
    figsize=FIGSIZE,
)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")


#################################  Bifurcation diagram of inferred landscape
FIGNAME = "phi2_bifs_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves, bifcolors = get_plnn_bifurcation_curves(
    model, num_starts=100, rng=rng
)
for curve, color in zip(bifcurves, bifcolors):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


# ##############################################################################
# ##############################################################################

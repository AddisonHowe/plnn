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


OUTDIR = "figures/out/fig3_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y + p1*x + p2*y

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches


##############################################################################
##############################################################################
##  Binary Choice Landscape

#################################  Heatmap of true landscape
FIGNAME = "phi1_heatmap"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

ax = plot_landscape(
    func_phi1_star, r=r, res=res, params=[0, 0], 
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
FIGNAME = "phi1_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_true, bifcolors_true = get_binary_choice_curves(rng=rng)
for curve, color in zip(bifcurves_true, bifcolors_true):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of true landscape in signals
FIGNAME = "phi1_bifs_signals"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    if len(curve) > 1:
        ax.plot(-curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Load the inferred landscape

MODELDIR = "data/trained_models/model_phi1_1a_v1_20240311_130654"

model, hps, idx, name, fpath = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR)

tilt_weights = model.get_parameters()['tilt.w'][0]
tilt_bias = model.get_parameters()['tilt.b'][0]

if tilt_bias is None:
    tilt_bias = np.zeros(tilt_weights.shape[0])
else:
    tilt_bias = tilt_bias[0]

def signal_to_tilts(signal):
    return np.dot(tilt_weights, signal) + tilt_bias

def tilts_to_signals(tilts):
    assert tilts.shape[0] == 2
    y = tilts - tilt_bias[:,None]
    return np.linalg.solve(tilt_weights, y)

print("tilt weights:\n", tilt_weights)
print("tilt bias:\n", tilt_bias)

#################################  Heatmap of inferred landscape

FIGNAME = "phi1_inferred"
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
FIGNAME = "phi1_bifs_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_inferred, bifcolors_inferred = get_plnn_bifurcation_curves(
    model, num_starts=100, rng=rng
)
for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of inferred landscape in signals
FIGNAME = "phi1_bifs_signals_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    if len(curve) > 1:
        curve_signal = tilts_to_signals(curve.T).T
        ax.plot(curve_signal[:,0], curve_signal[:,1], '-', color=color)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

# ax.set_xlim()
# ax.set_ylim()

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


# #################################  Combined bif diagram
FIGNAME = "phi1_combined_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    if len(curve) > 1:
        ax.plot(curve[:,0], curve[:,1], '-', color=color)

for curve, color in zip(bifcurves_true, bifcolors_true):
    if len(curve) > 1:
        ax.plot(curve[:,0], curve[:,1], '--', color=color)

ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

ax.set_xlim([-2, 2])
ax.set_ylim([-1, 3])

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


# #################################  Combined bif diagram in signals
FIGNAME = "phi1_combined_bifs_signals"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    if len(curve) > 1:
        curve_signal = tilts_to_signals(curve.T).T
        ax.plot(curve_signal[:,0], curve_signal[:,1], '-', color=color)

for curve, color in zip(bifcurves_true, bifcolors_true):
    if len(curve) > 1:
        ax.plot(-curve[:,0], curve[:,1], '--', color=color)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

ax.set_xlim([-2, 2])
ax.set_ylim([-1, 5])

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')



# # ##############################################################################
# # ##############################################################################

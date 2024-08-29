"""Figure 3b Script

Generate plots corresponding to those used in Figure 3 of the accompanying 
manuscript, but using the binary flip landscape instead of the binary choice.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.style.use('figures/styles/fig3.mplstyle')

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.pl import plot_landscape, plot_phi, plot_sigma_history
from plnn.pl import CHIR_COLOR, FGF_COLOR
from plnn.vectorfields import estimate_minima
from plnn.helpers import get_phi2_fixed_points

from cont.binary_flip import get_binary_flip_curves
from cont.plnn_bifurcations import get_plnn_bifurcation_curves 

SEED = 1234512345

rng = np.random.default_rng(seed=SEED)

MODELDIR = "data/trained_models/plnn_synbindec/model_phi2_1a_v_mmd1_20240704_142345"
SIGMA_TRUE = 0.3

OUTDIR = "figures/out/fig3b_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches

FP_MARKERS = {
    'saddle': 'x',
    'minimum': '*',
    'maximum': 'o',
}

##############################################################################
##############################################################################
##  Binary Flip Landscape

TILT_TO_PLOT = [-0.25, 0.]

#################################  Heatmap of true landscape
FIGNAME = "phi2_heatmap"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 100   # resolution
lognormalize = True
clip = None

fps, fp_types, fp_colors = get_phi2_fixed_points([TILT_TO_PLOT])

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=TILT_TO_PLOT, 
    lognormalize=lognormalize,
    clip=clip,
    title="True landscape",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=None,
    figsize=FIGSIZE,
    show=True,
)
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=FP_MARKERS[fp_type],
        markersize=2,
    )

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)
plt.close()

#################################  Bifurcation diagram of true landscape
FIGNAME = "phi2_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_true, bifcolors_true = get_binary_flip_curves(
    rng=rng, 
    add_flip_curves=True,
)

for curve, color in zip(bifcurves_true, bifcolors_true):
    linestyle = ':' if color == 'purple' else '-'
    linewidth = 1 if color == 'grey' else None
    ax.plot(curve[:,0], curve[:,1], linestyle=linestyle, linewidth=linewidth, 
            color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of true landscape in signals
FIGNAME = "phi2_bifs_signals"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    if len(curve) > 1:
        linestyle = ':' if color == 'purple' else '-'
        linewidth = 1 if color == 'grey' else None
        ax.plot(
            curve[:,0], curve[:,1], linestyle=linestyle, linewidth=linewidth, 
            color=color
        )

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Load the inferred landscape

model, hps, idx, name, fpath = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR)

tilt_weights = model.get_parameters()['tilt.w'][0]
tilt_bias = model.get_parameters()['tilt.b'][0]
noise_parameter = model.get_sigma()

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
print("inferred noise:\n", noise_parameter)

#################################  Heatmap of inferred landscape

FIGNAME = "phi2_inferred"
FIGSIZE = (5*sf, 5*sf)

r = 4       # box radius
res = 100   # resolution
lognormalize = True
clip = None

ax = plot_phi(
    model, tilt=TILT_TO_PLOT, 
    r=r, res=res,
    lognormalize=lognormalize,
    clip=clip,
    title="Inferred",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=None,
    figsize=FIGSIZE,
    show=True,
)

mins = estimate_minima(
    model, TILT_TO_PLOT, 
    n=50, 
    tol=1e-2,
    x0_range=[[-3, 3],[-3, 3]], 
    rng=rng,
)
print(mins)

for m in mins:
    ax.plot(m[0], m[1], marker='.', color='y', markersize=3)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)
plt.close()


#################################  Bifurcation diagram of inferred landscape
FIGNAME = "phi2_bifs_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_inferred, bifcolors_inferred, aux_info = get_plnn_bifurcation_curves(
    model, 
    num_starts=100, 
    maxiter=1000,
    ds=1e-3,
    min_ds=1e-8,
    max_ds=1e-2,
    max_delta_p=1e-1,
    rho=1e-2,
    return_aux_info=True,
    rng=rng,
    verbosity=0,
)

# Filter out singleton bifurcation curves and remove initial estimate point
keepidxs = [i for i in range(len(bifcurves_inferred)) if len(bifcurves_inferred[i]) > 1]
bifcurves_inferred = [bc[1:] for bc in bifcurves_inferred if len(bc) > 1]
bifcolors_inferred = [bifcolors_inferred[i] for i in keepidxs]

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of inferred landscape in signals
FIGNAME = "phi2_bifs_signals_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    curve_signal = tilts_to_signals(curve.T).T
    ax.plot(curve_signal[:,0], curve_signal[:,1], '-', color=color)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Combined bif diagram
FIGNAME = "phi2_combined_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    linewidth = 1 if color == 'grey' else None
    true_line, = ax.plot(
        curve[:,0], curve[:,1], '--', linewidth=linewidth, color=color
    )

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    inf_line, = ax.plot(curve[:,0], curve[:,1], '-', color=color, alpha=0.7)

ax.legend(
    [true_line, inf_line], ['true', 'inferred'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',
    fontsize='x-small'
)

ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

ax.set_xlim([-2, 3])
ax.set_ylim([-2, 2])

ax.set_title("Bifurcations (tilt space)")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)


#################################  Combined bif diagram in signals
FIGNAME = "phi2_combined_bifs_signals"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    linewidth = 1 if color == 'grey' else None
    true_line, = ax.plot(
        curve[:,0], curve[:,1], '--', linewidth=linewidth, color=color
    )

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    curve_signal = tilts_to_signals(curve.T).T
    inf_line, = ax.plot(
        curve_signal[:,0], curve_signal[:,1], '-', color=color, alpha=0.7
    )

ax.legend(
    [true_line, inf_line], ['true', 'inferred'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',        
    fontsize='x-small'
)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

ax.set_xlim([-3, 5])
ax.set_ylim([-4, 4])

ax.set_title("Bifurcations (signal space)")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)


##################################  Noise History
FIGNAME = "noise_history"
FIGSIZE = (5*sf, 3*sf)

logged_args, run_dict = load_model_training_metadata(MODELDIR)
sigma_hist = run_dict['sigma_hist']

ax = plot_sigma_history(
    sigma_hist, log=False, sigma_true=SIGMA_TRUE,
    color='k', marker=None, linestyle='-',
    figsize=FIGSIZE,
)
ax.axvline(
    idx, 0, 1,
    linestyle='--', 
    label=f"Inferred $\sigma={model.get_sigma():.3g}$"
)
ax.legend(fontsize='xx-small')
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)


##################################  Signal mapping
FIGNAME = "signal_mapping"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.set_aspect('equal')

scale = np.max(np.abs(tilt_weights))

ax.arrow(
    0, 0, -tilt_weights[0,0], -tilt_weights[1,0], 
    width=0.01*scale, 
    fc=CHIR_COLOR, ec=CHIR_COLOR,
    label="$s_1$"
)

ax.arrow(
    0, 0, -tilt_weights[0,1], -tilt_weights[1,1], 
    width=0.01*scale, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    label="$s_2$"
)

ax.set_title("Inferred signal effect")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-scale*1.1, scale*1.1])
ax.set_ylim([-scale*1.1, scale*1.1])

ax.legend(fontsize='xx-small')

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)


##############################################################################
##############################################################################

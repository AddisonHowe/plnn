"""Figure 3 Script (Synthetic Training)

Generate plots used in Figure 3 of the accompanying manuscript.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
plt.style.use('figures/manuscript/styles/fig_standard.mplstyle')
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.pl import plot_landscape, plot_phi, plot_sigma_history, plot_loss_history
from plnn.pl import CHIR_COLOR, FGF_COLOR
from plnn.vectorfields import estimate_minima
from plnn.helpers import get_phi1_fixed_points

from cont.binary_choice import get_binary_choice_curves
from cont.plnn_bifurcations import get_plnn_bifurcation_curves 


OUTDIR = "figures/manuscript/out/fig3_synthetic_training"
SAVEPLOTS = True
MODELDIR = "data/trained_models/plnn_synbindec/model_phi1_1a_v_mmd1_20240704_134102"
SIGMA_TRUE = 0.1
SEED = 123

LEGEND_FONTSIZE = 6
INSET_SCALE = "30%"


os.makedirs(OUTDIR, exist_ok=True)

rng = np.random.default_rng(seed=SEED)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches

FP_MARKERS = {
    'saddle': 'x',
    'minimum': 'o',
    'maximum': '^',
}


##############################################################################
##############################################################################
##  Load the inferred landscape

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


##############################################################################
##############################################################################
##  Binary Choice Landscape

TILT_TO_PLOT = [0., 0.5]

#################################  Heatmap of true landscape
FIGNAME = "phi1_heatmap"
FIGSIZE = (5.5*sf, 5.5*sf)

r = 2.5       # box radius
res = 50   # resolution
lognormalize = True
clip = None

fps, fp_types, fp_colors = get_phi1_fixed_points([TILT_TO_PLOT])

ax = plot_landscape(
    func_phi1_star, r=r, res=res, params=TILT_TO_PLOT, 
    lognormalize=lognormalize,
    clip=clip,
    title="Ground truth",
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    # cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    cbar_title="",
    equal_axes=True,
    saveas=None,
    figsize=FIGSIZE,
    show=True,
)
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=0.2 if marker == 'o' else 0.5,
    )

# ax.set_xticks([])
# ax.set_yticks([])

# Plot signal effect inset
subax = inset_axes(ax,
    width=INSET_SCALE,
    height=INSET_SCALE,
    loc=3,
    # bbox_to_anchor=(0, 0, 1, 1),
    # bbox_transform=ax.transAxes,
)
subax.set_aspect('equal')
subax.axis('off')
scale = 1.0
subax.arrow(
    0, 0, -1, 0, 
    width=0.01*scale, 
    fc=CHIR_COLOR, ec=CHIR_COLOR,
    label="$s_1$"
)
subax.arrow(
    0, 0, 0, -1, 
    width=0.01*scale, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    label="$s_2$"
)
subax.set_xlim([-scale*1.2, scale*1.2])
subax.set_ylim([-scale*1.2, scale*1.2])

plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()

#################################  Bifurcation diagram of true landscape
FIGNAME = "phi1_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_true, bifcolors_true = get_binary_choice_curves(
    rng=rng,
)
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
        ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Heatmap of inferred landscape

FIGNAME = "phi1_inferred"
FIGSIZE = (5.5*sf, 5.5*sf)

r = 2.5     # box radius
res = 100   # resolution
lognormalize = True
clip = None

ax = plot_phi(
    model, tilt=TILT_TO_PLOT, 
    r=r, res=res,
    lognormalize=lognormalize,
    clip=clip,
    title="Inferred",
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    # cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    cbar_title="",
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
for m in mins:
    ax.plot(m[0], m[1], marker='.', color='y', markersize=3)

# Plot signal effect inset
subax = inset_axes(ax,
    width=INSET_SCALE,
    height=INSET_SCALE,
    loc=3,
    # bbox_to_anchor=(0, 0, 1, 1),
    # bbox_transform=ax.transAxes,
)
subax.set_aspect('equal')
subax.axis('off')
scale = np.max(np.abs(tilt_weights))
subax.arrow(
    0, 0, -tilt_weights[0,0], -tilt_weights[1,0], 
    width=0.01*scale, 
    fc=CHIR_COLOR, ec=CHIR_COLOR,
    label="$s_1$"
)
subax.arrow(
    0, 0, -tilt_weights[0,1], -tilt_weights[1,1], 
    width=0.01*scale, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    label="$s_2$"
)
subax.set_xlim([-scale*1.2, scale*1.2])
subax.set_ylim([-scale*1.2, scale*1.2])

plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()

#################################  Bifurcation diagram of inferred landscape
FIGNAME = "phi1_bifs_inferred"
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
    rho=1e-1,
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
ax.set_ylim(-1, 3)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of inferred landscape in signals
FIGNAME = "phi1_bifs_signals_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    curve_signal = tilts_to_signals(curve.T).T
    ax.plot(curve_signal[:,0], curve_signal[:,1], '-', color=color)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##################################  Combined bif diagram
FIGNAME = "phi1_combined_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    true_line, = ax.plot(curve[:,0], curve[:,1], '--', color=color)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    inf_line, = ax.plot(curve[:,0], curve[:,1], '-', color=color, alpha=0.9)

ax.legend(
    [true_line, inf_line], ['ground truth', 'inferred'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',
    fontsize=LEGEND_FONTSIZE
)

ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

ax.set_xlim([-2, 2])
ax.set_ylim([-1, 3])

ax.set_title("Bifurcation diagram")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)


##################################  Combined bif diagram in signals
FIGNAME = "phi1_combined_bifs_signals"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    true_line, = ax.plot(curve[:,0], curve[:,1], '--', color=color)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    curve_signal = tilts_to_signals(curve.T).T
    inf_line, = ax.plot(curve_signal[:,0], curve_signal[:,1], '-', 
                        color=color, alpha=0.9)

ax.legend(
    [true_line, inf_line], ['ground truth', 'inferred'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',
    fontsize=LEGEND_FONTSIZE
)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

ax.set_xlim([-2, 2])
ax.set_ylim([-1, 5])

ax.set_title("Bifurcation diagram")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)


##################################  Noise History
FIGNAME = "noise_history"
FIGSIZE = (5.5*sf, 3.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

logged_args, run_dict = load_model_training_metadata(MODELDIR)
sigma_hist = run_dict['sigma_hist']

plot_sigma_history(
    sigma_hist, log=False, sigma_true=SIGMA_TRUE,
    color='k', marker=None, linestyle='-',
    title="", sigma_true_legend_label=f'ground truth $\\sigma^*={SIGMA_TRUE:.3g}$',
    figsize=FIGSIZE,
    ax=ax,
)
ylims = ax.get_ylim()
ax.vlines(
    idx, ylims[0], sigma_hist[idx],
    linestyles='--', colors='grey', linewidth=1, zorder=1, 
    label=f"inferred $\sigma={model.get_sigma():.2g}$"
)
ax.set_ylabel("noise $\sigma$")
ax.set_ylim(*ylims)
ax.legend(fontsize=LEGEND_FONTSIZE)
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()


##################################  Loss History
FIGNAME = "loss_history"
FIGSIZE = (5.5*sf, 3.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')
plot_loss_history(
    training_info['loss_hist_train'],
    training_info['loss_hist_valid'],
    log=True,
    color_train='r', color_valid='b',
    marker_train=None, marker_valid=None,
    linestyle_train='-', linestyle_valid='-',
    linewidth_train=1, linewidth_valid=1,
    alpha_train=0.7, alpha_valid=0.6,
    ax=ax
)
ax.set_xlabel("epoch")
ax.set_ylabel("error")
ax.set_title("")

ax.axvline(
    idx, 0, 1,
    linestyle='--', color='grey', linewidth=1, zorder=1,
)
ax.legend(fontsize=LEGEND_FONTSIZE)
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()

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

ax.legend(fontsize=LEGEND_FONTSIZE)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)


##################################  Signal prior
FIGNAME = "signal_prior"
FIGSIZE = (5*sf, 5*sf)

ALPHA = 0.5
COL0 = 'orange'
COL1 = 'purple'

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    true_line, = ax.plot(curve[:,0], curve[:,1], '--', color=color)


info_dict = {
    's10': [-0.50, 0.50],
    's20': [ 0.50, 1.50],
    's11': [-1.00, 1.00],
    's21': [-0.50, 0.50],
}

# Plot the prior for initial signal values
x0 = info_dict['s10'][0]
y0 = info_dict['s20'][0]
w = info_dict['s10'][1] - info_dict['s10'][0]
h = info_dict['s20'][1] - info_dict['s20'][0]
# Create the rectangle patch with transparency (alpha)
rectangle = patches.Rectangle(
    (x0, y0), w, h, 
    alpha=ALPHA, color=COL0, fill=None, hatch=4*'/',
)
p1 = ax.add_patch(rectangle)

# Plot the prior for final signal values
x0 = info_dict['s11'][0]
y0 = info_dict['s21'][0]
w = info_dict['s11'][1] - info_dict['s11'][0]
h = info_dict['s21'][1] - info_dict['s21'][0]
# Create the rectangle patch with transparency (alpha)
rectangle = patches.Rectangle(
    (x0, y0), w, h, 
    alpha=ALPHA, color=COL1, fill=None, hatch=4*'\\',
)
p2 = ax.add_patch(rectangle)

ax.legend(
    [p1, p2], ['initial', 'final'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',
    fontsize=LEGEND_FONTSIZE,
)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

ax.set_xlim([-2, 2])
ax.set_ylim([-1, 5])

ax.set_title("Signal prior")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)

##############################################################################
##############################################################################

"""Figure 5 Script

Generate plots used in Figure 5 of the accompanying manuscript, the results of
the first binary decision captured in the FACS data.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

import matplotlib.pyplot as plt
plt.style.use('figures/styles/fig5.mplstyle')

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.pl import plot_landscape, plot_phi
from plnn.vectorfields import estimate_minima

from cont.plnn_bifurcations import get_plnn_bifurcation_curves 

SEED = 123

rng = np.random.default_rng(seed=SEED)


OUTDIR = "figures/out/fig5_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches

MINMARKERSIZE = 3
MINMARKER = '.'
MINCOLOR = 'r'

##############################################################################
##############################################################################
##  Load the inferred landscape

MODELDIR = "data/trained_models/facs/model_facs_dec1b_2dpca_v1_20240624_133245"

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

FIGNAME = "dec1_phi_inferred"
FIGSIZE = (5*sf, 5*sf)

SIGNALS_TO_PLOT = [
    [0., 0.],
    [0., 1.],
    [0., 0.9],
    [1., 0.],
    [1., 1.],
    [1., 0.9],
]

r = 8       # box radius
res = 200   # resolution
lognormalize = True
clip = None

for i, sig_to_plot in enumerate(SIGNALS_TO_PLOT):
    ax = plot_phi(
        model, signal=sig_to_plot, 
        r=r, res=res,
        lognormalize=lognormalize,
        clip=clip,
        title=f"CHIR: {sig_to_plot[0]:.1f}, FGF: {sig_to_plot[1]:.1f}",
        ncontours=10,
        contour_linewidth=0.5,
        include_cbar=True,
        cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
        equal_axes=True,
        saveas=None,
        show=True,
        figsize=FIGSIZE,
    )

    mins = estimate_minima(
        model, model.tilt_module(np.array(sig_to_plot)), 
        n=50, 
        tol=1e-2,
        x0_range=[[-8, 8],[-8, 8]], 
        rng=rng,
    )
    print(mins)

    for m in mins:
        ax.plot(
            m[0], m[1], 
            marker=MINMARKER, 
            markersize=MINMARKERSIZE,
            color=MINCOLOR, 
        )

    plt.tight_layout()
    plt.savefig(
        f"{OUTDIR}/{FIGNAME}_{i}", bbox_inches='tight', 
        transparent=True
    )
    plt.close()

#################################  Bifurcation diagram of inferred landscape
FIGNAME = "dec1_phi_bifs_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_inferred, bifcolors_inferred = get_plnn_bifurcation_curves(
    model, 
    xlims=[-8, 8],
    ylims=[-8, 8],
    p1lims=[-8, 8],
    p2lims=[-8, 8],
    num_starts=100, 
    maxiter=1000,
    ds=1e-3,
    min_ds=1e-8,
    max_ds=1e-2,
    max_delta_p=1e-2,
    rho=1e-1,
    rng=rng,
    verbosity=1
)
for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    if len(curve) > 1:
        ax.plot(
            curve[:,0], curve[:,1], '-', color=color, linewidth=1,
        )

# ax.set_xlim(-8, 8)
# ax.set_ylim(-8, 8)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of inferred landscape in signals
FIGNAME = "dec1_phi_bifs_signals_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    if len(curve) > 1:
        curve_signal = tilts_to_signals(curve.T).T
        ax.plot(
            curve_signal[:,0], curve_signal[:,1], '-', color=color, linewidth=1,
        )

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

# ax.set_xlim()
# ax.set_ylim()

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##############################################################################
##############################################################################

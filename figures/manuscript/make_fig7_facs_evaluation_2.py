"""Figure 7 Script (FACS Evaluation)

Generate plots used in Figure 7 of the accompanying manuscript.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jrandom

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
plt.style.use('figures/manuscript/styles/fig_standard.mplstyle')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plnn.models import DeepPhiPLNN
from plnn.dataset import get_dataloaders
from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.vectorfields import estimate_minima
from plnn.pl import plot_landscape, plot_phi, plot_sigma_history 
from plnn.pl import plot_loss_history
from plnn.pl import CHIR_COLOR, FGF_COLOR

from cont.plnn_bifurcations import get_plnn_bifurcation_curves 

SEED = 12482582

rng = np.random.default_rng(seed=SEED)

MODELDIR = "data/trained_models/facs/model_facs_v3_dec1b_2dpca_v12b_20240719_005108"
DATDIR = "data/training_data/facs_v3/pca/dec1_fitonsubset/transition1_subset_epi_tr_ce_an_pc12"

OUTDIR = "figures/manuscript/out/fig7_facs_evaluation"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches

MINMARKERSIZE = 1
MINMARKER = '.'
MINCOLOR = 'y'

INSET_SCALE = "40%"
N_EST_MIN = 100

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

SIG1LIMS = [-1, 2]
SIG2LIMS = [-1, 2]

extremes = np.array([
    signal_to_tilts([SIG1LIMS[0], SIG2LIMS[0]]),
    signal_to_tilts([SIG1LIMS[0], SIG2LIMS[1]]),
    signal_to_tilts([SIG1LIMS[1], SIG2LIMS[0]]),
    signal_to_tilts([SIG1LIMS[1], SIG2LIMS[1]]),
])
P1LIMS = [extremes[:,0].min(), extremes[:,0].max()]
P2LIMS = [extremes[:,1].min(), extremes[:,1].max()]

##############################################################################
##  Load the training/validation/testing data

ncells_sample = logged_args.get('ncells_sample', 0)
length_multiplier = logged_args.get('passes_per_epoch', 1)

datdir_train = f"{DATDIR}/training"
datdir_valid = f"{DATDIR}/validation"
datdir_test = f"{DATDIR}/testing"
nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)
train_loader, _, test_loader, train_dset, valid_dset, test_dset = get_dataloaders(
    datdir_train, datdir_valid, nsims_train, nsims_valid,
    shuffle_train=False,
    shuffle_valid=False,
    return_datasets=True,
    include_test_data=True,
    datdir_test=datdir_test, nsims_test=nsims_test, shuffle_test=False,
    batch_size_test=1,
    ncells_sample=ncells_sample,
    length_multiplier=length_multiplier,
    seed=rng.integers(2**32)
)

print("Loaded datasets using parameters:")
print("\tncells_sample:", ncells_sample)
print("\tlength_multiplier:", length_multiplier)

xdata = np.vstack([d[0][1] for d in train_dset] + [train_dset[-1][1]])
XYBUFFER = 0.1
XMIN, YMIN = xdata.min(axis=0)
XMAX, YMAX = xdata.max(axis=0)

XMIN, XMAX = XMIN - (XMAX - XMIN) * XYBUFFER, XMAX + (XMAX - XMIN) * XYBUFFER
YMIN, YMAX = YMIN - (YMAX - YMIN) * XYBUFFER, YMAX + (YMAX - YMIN) * XYBUFFER
print("x range:", XMIN, XMAX)
print("y range:", YMIN, YMAX)


#################################  Loss history
FIGNAME = "loss_history"
FIGSIZE = (7*sf, 4*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
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
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")

ax.get_yaxis().get_major_formatter().labelOnlyBase = False
ax.set_yticks([0.2, 0.4, 0.6, 0.8], minor=True)
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(ticker.ScalarFormatter())

plt.savefig(
    f"{OUTDIR}/{FIGNAME}", transparent=True
)
plt.close()


#################################  Main Heatmap Diagram
FIGNAME = "phi_inferred_main"
FIGSIZE = (8*sf, 4*sf)

SIG_TO_PLOT = [0.0, 1.0]
PLOT_XLIMS = [-6, 5]
PLOT_YLIMS = [-3, 5]

res = 100   # resolution
lognormalize = True
clip = None

ax = plot_phi(
    model, signal=SIG_TO_PLOT, 
    xrange=PLOT_XLIMS,
    yrange=PLOT_YLIMS,
    res=res,
    lognormalize=lognormalize,
    clip=clip,
    title=f"CHIR: {SIG_TO_PLOT[0]:.1f}, FGF: {SIG_TO_PLOT[1]:.1f}",
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    xlabel=None,
    ylabel=None,
    equal_axes=True,
    saveas=None,
    show=True,
    figsize=FIGSIZE,
)

mins = estimate_minima(
    model, model.tilt_module(np.array(SIG_TO_PLOT)), 
    n=N_EST_MIN, 
    tol=1e-3,
    x0_range=[[XMIN, XMAX],[YMIN, YMAX]], 
    rng=rng,
)

for m in mins:
    ax.plot(
        m[0], m[1], 
        marker=MINMARKER,  
        markersize=MINMARKERSIZE,
        color=MINCOLOR, 
    )

plt.savefig(
    f"{OUTDIR}/{FIGNAME}", bbox_inches="tight", 
    transparent=True
)
plt.close()



#################################  Heatmaps of inferred landscape

FIGNAME = "phi_inferred"
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
res = 100   # resolution
lognormalize = True
clip = None

for i, sig_to_plot in enumerate(SIGNALS_TO_PLOT):
    ax = plot_phi(
        model, signal=sig_to_plot, 
        xrange=PLOT_XLIMS,  #[XMIN, XMAX]
        yrange=PLOT_YLIMS,  #[YMIN, YMAX]
        res=res,
        lognormalize=lognormalize,
        clip=clip,
        title=f"CHIR: {sig_to_plot[0]:.1f}, FGF: {sig_to_plot[1]:.1f}",
        ncontours=10,
        contour_linewidth=0.5,
        contour_linealpha=0.5,
        include_cbar=True,
        cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
        xlabel=None,
        ylabel=None,
        equal_axes=True,
        saveas=None,
        show=True,
        figsize=FIGSIZE,
    )

    # Plot signal effect inset
    subax = inset_axes(ax,
        width=INSET_SCALE,
        height=INSET_SCALE,
        loc='lower right',
        bbox_to_anchor=(0.07, -0.1, 1, 1),
        bbox_transform=ax.transAxes,
    )
    subax.set_aspect('equal')
    subax.axis('off')
    scale = np.max(np.abs(tilt_weights))
    arr1 = subax.arrow(
        0, 0, -tilt_weights[0,0], -tilt_weights[1,0], 
        width=0.01*scale, 
        fc=CHIR_COLOR, ec=CHIR_COLOR,
        label="CHIR"
    )
    # subax.annotate(
    #     "CHIR", (-tilt_weights[0,0], -tilt_weights[1,0]), 
    #     ha='right', va='bottom', 
    #     fontsize=6, 
    # )
     
    arr2 = subax.arrow(
        0, 0, -tilt_weights[0,1], -tilt_weights[1,1], 
        width=0.01*scale, 
        fc=FGF_COLOR, ec=FGF_COLOR,
        label="FGF"
    )
    # subax.annotate(
    #     "FGF", (-tilt_weights[0,1], -tilt_weights[1,1]), 
    #     ha='right', va='bottom', 
    #     fontsize=6, 
    # )
    
    subax.set_xlim([-scale*1.2, scale*1.2])
    subax.set_ylim([-scale*1.2, scale*1.2])

    # Find and plot minima
    mins = estimate_minima(
        model, model.tilt_module(np.array(sig_to_plot)), 
        n=N_EST_MIN, 
        tol=1e-3,
        x0_range=[[XMIN, XMAX],[YMIN, YMAX]], 
        rng=rng,
    )

    for m in mins:
        ax.plot(
            m[0], m[1], 
            marker=MINMARKER,  
            markersize=MINMARKERSIZE,
            color=MINCOLOR, 
        )

    # plt.tight_layout()
    plt.savefig(
        f"{OUTDIR}/{FIGNAME}_{i}", #bbox_inches='tight', 
        transparent=True
    )
    plt.close()

#################################  Bifurcation diagram of inferred landscape
FIGNAME = "phi_bifs_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_inferred, bifcolors_inferred = get_plnn_bifurcation_curves(
    model, 
    xlims=[XMIN, XMAX],
    ylims=[YMIN, YMAX],
    p1lims=P1LIMS,
    p2lims=P2LIMS,
    num_starts=100, 
    maxiter=1000,
    ds=1e-3,
    min_ds=1e-8,
    max_ds=1e-2,
    max_delta_p=1e-2,
    rho=1e-1,
    rng=rng,
    verbosity=0
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
FIGNAME = "phi_bifs_signals_inferred"
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

ax.set_xlim(*SIG1LIMS)
ax.set_ylim(*SIG2LIMS)

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of inferred landscape in signals
FIGNAME = "signal_direction"
FIGSIZE = (3*sf, 3*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.set_aspect('equal')

scale = np.max(np.abs(tilt_weights))

ax.arrow(
    0, 0, -tilt_weights[0,0], -tilt_weights[1,0], 
    width=0.01*scale, 
    fc=CHIR_COLOR, ec=CHIR_COLOR,
    label="CHIR"
)

ax.arrow(
    0, 0, -tilt_weights[0,1], -tilt_weights[1,1], 
    width=0.01*scale, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    label="FGF"
)

# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
ax.set_xlim([-scale*1.1, scale*1.1])
ax.set_ylim([-scale*1.1, scale*1.1])

ax.legend(fontsize='xx-small');

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)


#################################  Simulation snapshots
FIGNAME = "simulation_snapshot"
FIGSIZE = (2.6*sf, 2.6*sf)

XLIMS = [-8.461315377018, 10.01259687722186]
YLIMS = [-3.8205954067127337, 8.523846462122545]


rate = train_dset[0][0][3][0,-1]
print("rate:", rate)
CONDITIONS = {
    'CHIR 2-4' : [[2.0, 1.0, 0.0, rate], [1.0, 1.0, 0.9, rate]],
    'CHIR 2-5 FGF 2-4' : [[5.0, 1.0, 0.0, rate], [2.0, 1.0, 0.0, rate]],
}

x0 = test_dset[0][0][1]
tfin = 1.5

key = jrandom.PRNGKey(seed=rng.integers(2**32))
for cond_name in CONDITIONS:
    sigparams = np.array(CONDITIONS[cond_name])
    key, subkey = jrandom.split(key, 2)
    ts_all, xs_all, sigs_all, _ = model.run_landscape_simulation(
        x0, tfin, [0.5], sigparams, subkey
    )
    sim_ts = ts_all[0]
    sim_xs = xs_all[0]
    sim_sigs = sigs_all[0]

    print(sim_ts.shape, sim_xs.shape)
    np.save(f"{OUTDIR}/simdata_{cond_name}_ts.npy", sim_ts)
    np.save(f"{OUTDIR}/simdata_{cond_name}_xs.npy", sim_xs)

    for i, t in enumerate(sim_ts):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')
        xs = sim_xs[i]
        sig = sim_sigs[i]

        plot_phi(
            model, signal=sig, 
            xrange=XLIMS,
            yrange=YLIMS,
            res=50,
            lognormalize=True,
            clip=None,
            title=None,
            ncontours=5,
            contour_linewidth=0.5,
            contour_linealpha=0.2,
            include_cbar=False,
            xlabel=None,
            ylabel=None,
            equal_axes=False,
            tight_layout=False,
            saveas=None,
            show=True,
            ax=ax
        )        

        ax.plot(
            xs[:,0], xs[:,1], '.',
            color='k',
            markersize=1,
            rasterized=False,
            alpha=0.3
        )
        ax.set_xlim(*XLIMS)
        ax.set_ylim(*YLIMS)
        ax.set_xticks([0, 10])
        ax.set_yticks([0, 5])
        # plt.tight_layout()
        plt.savefig(
            f"{OUTDIR}/{FIGNAME}_{cond_name}_d{t}.pdf", 
            transparent=True,
        )
        plt.close()


##############################################################################
##############################################################################

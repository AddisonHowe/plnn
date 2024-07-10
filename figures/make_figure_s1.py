"""Figure S1 Script

Generate plots used in Figure S1 of the accompanying manuscript.
"""

import argparse
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.style.use('figures/styles/fig_standard.mplstyle')

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.pl import plot_landscape, plot_phi
from plnn.pl import plot_loss_history, plot_learning_rate_history
from plnn.pl import plot_sigma_history

from cont.binary_choice import get_binary_choice_curves
from cont.plnn_bifurcations import get_plnn_bifurcation_curves 


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, 
                    help="Name of trained model directory w/o prefix.")
parser.add_argument('--truesigma', type=float, required=True)
parser.add_argument('--logloss', default=True, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--startidx', default=0, type=int)
args = parser.parse_args()

modeldir = args.input  # trained model directory

logloss = args.logloss
startidx = args.startidx
truesigma = args.truesigma
print("logloss", logloss)
print("startidx", startidx)
print("truesigma", truesigma)

SAVEPLOTS = True

MODELDIRBASE = "data/trained_models"
OUTDIRBASE = "figures/out/fig_S1"
SEED = 12345
rng = np.random.default_rng(seed=SEED)

OUTDIR = f"{OUTDIRBASE}/{modeldir}/"

os.makedirs(OUTDIR, exist_ok=True)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches


COLOR_TRAIN = 'r'
COLOR_VALID = 'b'
MARKER_TRAIN = None
MARKER_VALID = None
LINESTYLE_TRAIN = '-'
LINESTYLE_VALID = '-'
LINEWIDTH_TRAIN = 1
LINEWIDTH_VALID = 1


##############################################################################
##############################################################################
##  Load model and training information

MODELDIR = f"{MODELDIRBASE}/{modeldir}"

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
##  Plot the training history

FIGNAME = "training_history"
FIGSIZE = (8*sf, 12*sf)

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE)

ax=axes[0]
plot_loss_history(
    training_info['loss_hist_train'],
    training_info['loss_hist_valid'],
    startidx=startidx, log=logloss, 
    color_train=COLOR_TRAIN, color_valid=COLOR_VALID,
    marker_train=MARKER_TRAIN, marker_valid=MARKER_VALID,
    linestyle_train=LINESTYLE_TRAIN, linestyle_valid=LINESTYLE_VALID,
    linewidth_train=LINEWIDTH_TRAIN, linewidth_valid=LINEWIDTH_VALID,
    alpha_train=0.7, alpha_valid=0.6,
    ax=ax
)
ax.set_xlabel("")

ax=axes[1]
plot_sigma_history(
    training_info['sigma_hist'],
    log=False, 
    sigma_true=truesigma,
    color='k',
    linewidth=2,
    marker=None,
    ax=ax
)
ax.set_xlabel("")

ax=axes[2]
plot_learning_rate_history(
    training_info['learning_rate_hist'],
    log=False, 
    ax=ax
)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##############################################################################
##############################################################################
##  Plot parameter history

FIGNAME = "parameter_history"
FIGSIZE = (8*sf, 8*sf)

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

ax11 = axes[0][0]
ax12 = axes[0][1]
ax21 = axes[1][0]
ax22 = axes[1][1]

color = 'k'
linewidth = 1


ax11.plot(
    training_info['tilt_weight_hist'][:,0,0,0],
    color=color,
    linewidth=linewidth,
    label="$A_{11}$"
)

ax12.plot(
    training_info['tilt_weight_hist'][:,0,0,1],
    color=color,
    linewidth=linewidth,
    label="$A_{12}$"
)

ax21.plot(
    training_info['tilt_weight_hist'][:,0,1,0],
    color=color,
    linewidth=linewidth,
    label="$A_{21}$"
)

ax22.plot(
    training_info['tilt_weight_hist'][:,0,1,1],
    color=color,
    linewidth=linewidth,
    label="$A_{22}$"
)

for ax in axes.flatten():
    ax.legend()

for ax in [ax21, ax22]:
    ax.set_xlabel("Epoch")

fig.suptitle("Signal transformation over training")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##############################################################################
##############################################################################
##  Plot NORMALIZED parameter history

FIGNAME = "parameter_history_difference"
FIGSIZE = (8*sf, 8*sf)

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

ax11 = axes[0][0]
ax12 = axes[0][1]
ax21 = axes[1][0]
ax22 = axes[1][1]

tilt_weights = training_info['tilt_weight_hist'][:,0,:,:]
tilt_weights_true = np.array([[1., 0.],[0., 1.]])

tilt_weights_diffs = tilt_weights - tilt_weights_true

color = 'k'
linewidth = 1


ax11.plot(
    tilt_weights_diffs[:,0,0],
    color=color,
    linewidth=linewidth,
    label="$A_{11} - A_{11}^*$"
)

ax12.plot(
    tilt_weights_diffs[:,0,1],
    color=color,
    linewidth=linewidth,
    label="$A_{12} - A_{12}^*$"
)

ax21.plot(
    tilt_weights_diffs[:,1,0],
    color=color,
    linewidth=linewidth,
    label="$A_{21} - A_{21}^*$"
)

ax22.plot(
    tilt_weights_diffs[:,1,1],
    color=color,
    linewidth=linewidth,
    label="$A_{22} - A_{22}^*$"
)

for ax in axes.flatten():
    ax.legend(loc='lower right')
    ax.axhline(0, 0, 1, color='k', alpha=0.5, linestyle='--', linewidth=1)

for ax in [ax21, ax22]:
    ax.set_xlabel("Epoch")

fig.suptitle("Signal transformation error")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')

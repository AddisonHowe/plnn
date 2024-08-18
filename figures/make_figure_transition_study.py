"""Figure 6 Script

Generate plots used in Figure 6 of the accompanying manuscript.
"""

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figures/styles/fig6.mplstyle')

from plnn.models import DeepPhiPLNN
from plnn.pl import plot_loss_history, plot_sigma_history

from cont.plnn_bifurcations import get_plnn_bifurcation_curves 

SEED = 31324
rng = np.random.default_rng(seed=SEED)

OUTDIR = "figures/out/fig6_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def collect_r_indices(directory, basename):
    fpaths = glob(f"{directory}/{basename}_r*", recursive=False)
    fnames = [f.replace(f"{directory}/{basename}_r", "") for f in fpaths]
    fnames = [s.split("_")[0] for s in fnames]
    ridxs = [int(r) for r in fnames]
    ridxs = np.sort(ridxs)
    return ridxs

sf = 1/2.54  # scale factor from [cm] to inches

##############################################################################
##############################################################################
##  Transition Study Results

FIGNAME = "transition_rate_study"
FIGSIZE = (12*sf,10*sf)
BUFFER_L_FACTOR = 0.05
BUFFER_R_FACTOR = 1.05
BUFFER_T = 0.2
BUFFER_B = 0.2
SHIFT_FACTOR = -0.5
INSET_WIDTH_FACTOR = 0.90
INSET_HEIGHT_RATIO = 1.
BOXR = 4
RES = 50
TWO_ROWS = True

run_and_data_ids = [
    ("transition_rate_study_model_training_kl1_1", "tr_study1"),
    ("transition_rate_study_model_training_kl1_1", "tr_study2"),
    ("transition_rate_study_model_training_kl1_1", "tr_study3"),
    ("transition_rate_study_model_training_kl1_1", "tr_study4"),
    ("transition_rate_study_model_training_kl1_1", "tr_study5"),
    ("transition_rate_study_model_training_kl1_1", "tr_study6"),
    # ("transition_rate_study_model_training_kl1_2", "tr_study1"),N
    # ("transition_rate_study_model_training_kl1_2", "tr_study2"),
    # ("transition_rate_study_model_training_kl1_2", "tr_study3"),
    # ("transition_rate_study_model_training_kl1_2", "tr_study4"),
    # ("transition_rate_study_model_training_kl1_2", "tr_study5"),
    # ("transition_rate_study_model_training_kl1_2", "tr_study6"),
    # ("transition_rate_study_model_training_kl1_fix_noise",  "tr_study1"),
    # ("transition_rate_study_model_training_kl1_fix_noise",  "tr_study2"),
    # ("transition_rate_study_model_training_kl1_fix_noise",  "tr_study3"),
    # ("transition_rate_study_model_training_kl1_fix_noise",  "tr_study4"),
    # ("transition_rate_study_model_training_kl1_fix_noise",  "tr_study5"),
    # ("transition_rate_study_model_training_kl1_fix_noise",  "tr_study6"),
    # ("transition_rate_study_model_training_kl2_1", "tr_study201"),
    # ("transition_rate_study_model_training_kl2_1", "tr_study202"),
    # ("transition_rate_study_model_training_kl2_1", "tr_study203"),
    # ("transition_rate_study_model_training_kl2_1", "tr_study204"),
    # ("transition_rate_study_model_training_kl2_1", "tr_study205"),
    # ("transition_rate_study_model_training_kl2",            "tr_study206"),
    # ("transition_rate_study_model_training_kl2_fix_noise",  "tr_study201"),
    # ("transition_rate_study_model_training_kl2_fix_noise",  "tr_study202"),
    # ("transition_rate_study_model_training_kl2_fix_noise",  "tr_study203"),
    # ("transition_rate_study_model_training_kl2_fix_noise",  "tr_study204"),
    # ("transition_rate_study_model_training_kl2_fix_noise",  "tr_study205"),
    # ("transition_rate_study_model_training_kl2_fix_noise",  "tr_study206"),
]

for run_id, data_id in run_and_data_ids:

    outsubdir = f"{OUTDIR}/{run_id}/{data_id}"  # Output directory
    os.makedirs(outsubdir, exist_ok=True)

    basedatdir = f"data/transition_rate_studies/{data_id}"
    basedatdir_train = f"{basedatdir}/{data_id}_training"
    basedatdir_valid = f"{basedatdir}/{data_id}_validation"

    resdir = f"data/transition_rate_study_results/{run_id}/{data_id}"

    ridxs = np.unique(collect_r_indices(resdir, f"model_{data_id}"))

    phi     = np.genfromtxt(f"{basedatdir_train}/landscape_name", dtype=str)
    sigma   = np.genfromtxt(f"{basedatdir_train}/sigma",          dtype=float)
    tfin    = np.genfromtxt(f"{basedatdir_train}/tfin",           dtype=float)
    dt      = np.genfromtxt(f"{basedatdir_train}/dt",             dtype=float)
    dt_save = np.genfromtxt(f"{basedatdir_train}/dt_save",        dtype=float)
    tfin    = np.genfromtxt(f"{basedatdir_train}/tfin",           dtype=float)
    ncells  = np.genfromtxt(f"{basedatdir_train}/ncells",         dtype=int)

    log_r_vals = []
    best_models = []

    for j, ridx in enumerate(ridxs):
        modeldir = glob(f"{resdir}/model_{data_id}_r{ridx}*", recursive=False)
        assert len(modeldir) == 1, "Multiple runs found!"
        modeldir = modeldir[0]
        print(modeldir)
        datdir_train = f"{basedatdir_train}/r{ridx}"
        datdir_valid = f"{basedatdir_valid}/r{ridx}"
        assert np.genfromtxt(f"{datdir_train}/ridx.txt", dtype=int) == ridx, \
            "r mismatch"
        assert np.genfromtxt(f"{datdir_valid}/ridx.txt", dtype=int) == ridx, \
            "r mismatch"

        logr = np.genfromtxt(f"{datdir_train}/logr.txt", dtype=float)
        r = np.exp(logr)
        nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
        nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
        
        log_r_vals.append(logr)
        
        # Load validation loss history to recover best model index
        valid_loss_hist = np.load(f"{modeldir}/validation_loss_history.npy")
        try:
            train_loss_hist = np.load(f"{modeldir}/training_loss_history.npy")
        except ValueError as e:
            print(f"{modeldir}/training_loss_history.npy")
        best_idx = 1 + np.argmin(valid_loss_hist)  # add 1 as no initial loss
        model_fpath = f"{modeldir}/states/model_{data_id}_{best_idx}.pth"
        model, hparams = DeepPhiPLNN.load(model_fpath, dtype=np.float64)
        best_models.append((model, best_idx, valid_loss_hist[-1]))

        # Plot validation and training histories
        plot_loss_history(
            train_loss_hist, valid_loss_hist, 
            startidx=0,
            log=True,
            title="Loss history",
            saveas=f"{outsubdir}/loss_history_r{ridx}.pdf"
        )
        plt.close()

        # Plot sigma history
        fig, ax = plt.subplots(1, 1)
        sigma_hist = np.load(f"{modeldir}/sigma_history.npy")
        plot_sigma_history(
            sigma_hist,
            log=False,
            title="$\sigma$ history",
            saveas=None,
            ax=ax,
        )
        ax.hlines(sigma, *ax.get_xlim(), label="True $\sigma$")
        ax.legend()
        plt.savefig(f"{outsubdir}/sigma_history_r{ridx}.pdf")
        plt.close()

        # Plot model
        model.plot_phi(
            r=BOXR, res=RES, 
            normalize=True,
            lognormalize=True,
            equal_axes=True,
            saveas=f"{outsubdir}/phi_inferred_r{ridx}.pdf",
            show=False,
        )

        # Plot bifurcation curves
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

        bifcurves_inferred, bifcolors_inferred = get_plnn_bifurcation_curves(
            model, num_starts=100, rng=rng
        )
        for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
            ax.plot(curve[:,0], curve[:,1], '.', color=color)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_xlabel("$\\tau_1$")
        ax.set_ylabel("$\\tau_2$")

        plt.savefig(f"{outsubdir}/bifs_r{ridx}.pdf", bbox_inches='tight')
        plt.close()


    log_r_vals = np.array(log_r_vals)

    ninsets = len(best_models)
    base_y_offset = -0.1
    if TWO_ROWS:
        inset_width = 2 * INSET_WIDTH_FACTOR * np.min(np.diff(log_r_vals))
        inset_height = inset_width * INSET_HEIGHT_RATIO
        y_offsets = [base_y_offset, inset_height + base_y_offset]
    else:
        inset_width = INSET_WIDTH_FACTOR * np.min(np.diff(log_r_vals))
        inset_height = inset_width * INSET_HEIGHT_RATIO
        y_offsets = [base_y_offset]
            
    dot_y_pos = base_y_offset * 0.5

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.set_aspect('equal', adjustable='box')
    print(fig.get_size_inches()/sf)
    plt.tight_layout()

    inset_x_positions = np.zeros(len(best_models))
    inset_y_positions = np.zeros(len(best_models))
    for i, (model, best_idx, loss) in enumerate(best_models):
        if TWO_ROWS and (i % 2 == 0):
            inset_x = log_r_vals[i] + inset_width * SHIFT_FACTOR
            inset_x_positions[i] = inset_x
            inset_y = 0.1 + inset_height + 0.05
            inset_y_positions[i] = inset_y
        else:
            inset_x = log_r_vals[i] + inset_width * SHIFT_FACTOR
            inset_x_positions[i] = inset_x
            inset_y = 0.1
            inset_y_positions[i] = inset_y
        
        inset = ax.inset_axes(
            [inset_x, inset_y, inset_width, inset_height], 
            transform=ax.transData, 
            xticks=[], yticks=[], xlabel=None, ylabel=None
        )
            
        model.plot_phi(
            r=BOXR, res=RES, 
            normalize=True,
            lognormalize=True,
            show=True,
            ax=inset,
            equal_axes=True,
            xlims=[-4, 4],
            ylims=[-4, 4],
            include_cbar=False,
            title=None, xlabel=None, ylabel=None,
            tight_layout=False,
        )

        rinv = 1/np.exp(log_r_vals[i])
        s = f"After {best_idx} epochs,\nValidation loss: {loss:.4g}"
        s += f"\n$r^{{-1}}={rinv:.4g}$"
        s += f"    $2/(r\Delta T)\\approx{2*rinv/dt_save:.3g}$"
        inset.text(0.02, 0.02, s, transform=inset.transAxes, fontsize=5)

    print(fig.get_size_inches()/sf)
    plt.box(False)
    print(fig.get_size_inches()/sf)
    ax.set_xlim(inset_x_positions.min() - BUFFER_L_FACTOR * inset_width, 
                inset_x_positions.max() + BUFFER_R_FACTOR * inset_width)
    
    ax.hlines(base_y_offset, *ax.get_xlim(), colors='k', linestyles='-')
    ax.plot(log_r_vals, dot_y_pos + np.zeros(log_r_vals.shape), 'ko')
    ax.plot(
        [log_r_vals, log_r_vals], 
        [dot_y_pos * np.ones(inset_y_positions.shape), inset_y_positions], 
        'k-'
    )

    ax.set_title(f"Inferred Models ($\Delta T={dt_save:.1f}$)")
    ax.set_xlabel(f"$\ln(r)$\n$\leftarrow$ (slower transition) $\leftarrow$")
    
    ax.set_xticks(log_r_vals, log_r_vals)
    ax.set_yticks([])

    print(fig.get_size_inches()/sf)

    plt.savefig(f"{outsubdir}/{FIGNAME}.pdf", bbox_inches="tight")
    plt.close()

##############################################################################
##############################################################################
    

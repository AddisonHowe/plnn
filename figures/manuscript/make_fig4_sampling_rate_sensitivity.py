"""Figure 4 Script (Sampling Rate Sensitivity)

Generate plots used in Figure 4 of the accompanying manuscript.
"""

import argparse
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
plt.style.use('figures/manuscript/styles/fig_standard.mplstyle')
from sklearn.neighbors import NearestNeighbors

from plnn.dataset import get_dataloaders
from plnn.io import load_model_from_directory
from plnn.helpers import get_hist2d


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type=str, required=True, 
                    choices=['phi1_2', 'phi1_3', 'phi1_4'])
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


VERSION = args.version
THRESHOLD = args.threshold
SEED = args.seed

OUTDIR = f"figures/manuscript/out/fig4_sampling_rate_sensitivity/{VERSION}"
SAVEPLOTS = True

if args.threshold is None:
    THRESHOLD = {
        'phi1_2': 1.0, 
        'phi1_3': 1.0, 
        'phi1_4': 1.0
    }[VERSION]

if args.seed == 0:
    SEED = {
        'phi1_2': 12314, 
        'phi1_3': 34847, 
        'phi1_4': 78378
    }[VERSION]

rng = np.random.default_rng(seed=SEED)


os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches

##############################################################################
##############################################################################

BASEMODELDIR = "data/trained_models/plnn_synbindec"

DT_TO_DATDIR = {
    5  : f"data/training_data/data_{VERSION}a",
    10 : f"data/training_data/data_{VERSION}b",
    20 : f"data/training_data/data_{VERSION}c",
}

if VERSION == 'phi1_2':
    DT_TO_MODELS = {
        5: [
            "model_phi1_2a_v_mmd1_20240807_171303",
            "model_phi1_2a_v_mmd1_20240813_193424",
            "model_phi1_2a_v_mmd1_20240813_194028",
            "model_phi1_2a_v_mmd1_20240813_194433",
        ],
        10: [
            "model_phi1_2b_v_mmd1_20240807_171303",
            "model_phi1_2b_v_mmd1_20240813_193441",
            "model_phi1_2b_v_mmd1_20240813_193832",
            "model_phi1_2b_v_mmd1_20240813_194359",
        ],
        20: [
            "model_phi1_2c_v_mmd1_20240807_171303",
            "model_phi1_2c_v_mmd1_20240813_193441",
            "model_phi1_2c_v_mmd1_20240813_193755",
            "model_phi1_2c_v_mmd1_20240813_194114",
        ]
    } 
elif VERSION == 'phi1_3':
    DT_TO_MODELS = {
        5: [
            "model_phi1_3a_v_mmd1_20240822_132757",
            "model_phi1_3a_v_mmd1_20240822_142013",
            "model_phi1_3a_v_mmd1_20240822_142221",
            "model_phi1_3a_v_mmd1_20240822_142352",
            "model_phi1_3a_v_mmd1_20240823_172003",
            "model_phi1_3a_v_mmd1_20240823_172105",
            "model_phi1_3a_v_mmd1_20240823_172338",
            "model_phi1_3a_v_mmd1_20240823_172634",
        ],
        10: [
            "model_phi1_3b_v_mmd1_20240822_132815",
            "model_phi1_3b_v_mmd1_20240822_142553",
            "model_phi1_3b_v_mmd1_20240822_152909",
            "model_phi1_3b_v_mmd1_20240822_153440",
            "model_phi1_3b_v_mmd1_20240823_172003",
            "model_phi1_3b_v_mmd1_20240823_172217",
            "model_phi1_3b_v_mmd1_20240823_172338",
            "model_phi1_3b_v_mmd1_20240823_172713",
            
        ],
        20: [
            "model_phi1_3c_v_mmd1_20240822_133026",
            "model_phi1_3c_v_mmd1_20240822_142221",
            "model_phi1_3c_v_mmd1_20240822_142314",
            "model_phi1_3c_v_mmd1_20240822_142622",
            "model_phi1_3c_v_mmd1_20240823_172003",
            "model_phi1_3c_v_mmd1_20240823_172231",
            "model_phi1_3c_v_mmd1_20240823_172434",
            "model_phi1_3c_v_mmd1_20240823_172912",
        ]
    }
elif VERSION == 'phi1_4':
    DT_TO_MODELS = {
        5: [
            "model_phi1_4a_v_mmd1_20240822_132805",
            "model_phi1_4a_v_mmd1_20240822_141949",
            "model_phi1_4a_v_mmd1_20240822_144331",
            "model_phi1_4a_v_mmd1_20240822_144616",
            "model_phi1_4a_v_mmd1_20240823_175457",
            "model_phi1_4a_v_mmd1_20240823_171624",
            "model_phi1_4a_v_mmd1_20240826_101447",
            "model_phi1_4a_v_mmd1_20240826_102013",
        ],
        10: [
            "model_phi1_4b_v_mmd1_20240822_132805",
            "model_phi1_4b_v_mmd1_20240822_144432",
            "model_phi1_4b_v_mmd1_20240822_144616",
            "model_phi1_4b_v_mmd1_20240822_144632",
            "model_phi1_4b_v_mmd1_20240823_171535",
            "model_phi1_4b_v_mmd1_20240823_171538",
            "model_phi1_4b_v_mmd1_20240823_171624",
            "model_phi1_4b_v_mmd1_20240823_182813",
        ],
        20: [
            "model_phi1_4c_v_mmd1_20240822_132805",
            "model_phi1_4c_v_mmd1_20240822_144644",
            "model_phi1_4c_v_mmd1_20240822_144956",
            "model_phi1_4c_v_mmd1_20240822_152345",
            "model_phi1_4c_v_mmd1_20240823_171535",
            "model_phi1_4c_v_mmd1_20240823_171538",
            "model_phi1_4c_v_mmd1_20240823_174127",
            "model_phi1_4c_v_mmd1_20240823_183416",
        ]
    }

dt_list = np.sort(list(DT_TO_DATDIR.keys()))


def get_models_sorted(dt):
    model_list = DT_TO_MODELS[dt]
    loaded_models = []
    loaded_epochs = []
    for m in model_list:
        modeldir = f"{BASEMODELDIR}/{m}"
        _, _, epoch_loaded, _, _ = load_model_from_directory(modeldir)
        loaded_models.append(m)
        loaded_epochs.append(epoch_loaded)
    idxs = np.argsort(loaded_epochs)
    loaded_models = [loaded_models[i] for i in idxs]
    loaded_epochs = [loaded_epochs[i] for i in idxs]
    return loaded_models, loaded_epochs

##############################################################################
##############################################################################
####  Mean loss with 2 std devs for each model trained.
FIGNAME = "sampling_rate_sensitivity"
FIGSIZE = (10*sf, 5*sf)

markers = ['^', 'v', '<', '>']
colors = ['green', 'orange', 'blue']
offset = 0.1

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for dt_idx, dt in enumerate(dt_list):
    datdir = DT_TO_DATDIR[dt]
    model_list, epoch_list = get_models_sorted(dt)

    num_models = len(model_list)
    for i, m in enumerate(model_list):
        loss_data_dir = f"{BASEMODELDIR}/{m}/eval"
        filelist = [
            f.removesuffix('.npy') 
            for f in os.listdir(loss_data_dir) 
            if f.endswith(".npy")
        ]
        filelist = [s.split('_') for s in filelist]
        dt0_values = [float(v[3]) for v in filelist]
        dt0 = np.min(dt0_values)
        losses = np.load(f"{loss_data_dir}/losses_test_dt_{dt0}.npy")
        nconds, ndata_per_cond, nresamps, nreps = losses.shape
        assert nresamps == 1, "nresamps should be 1"
        losses = losses.squeeze(2)
        assert losses.shape == (nconds, ndata_per_cond, nreps), "bad loss shape"
        mean_loss = losses.mean()
        std_loss = losses.std()
        xpos = 1 + dt_idx + i*offset - num_models/2*offset
        if num_models % 2 == 0:
            xpos += offset / 2
        ax.errorbar(
            xpos, mean_loss, marker='.',
            yerr=2*std_loss, 
            capsize=3, 
            linestyle="None",
            color=colors[dt_idx],
        )

ax.set_xlim(ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset)
ax.set_xticks(range(1, len(dt_list) + 1), dt_list)
ax.set_ylabel("loss")

plt.savefig(f"{OUTDIR}/{FIGNAME}.pdf")


##############################################################################
##############################################################################
####  Box and whisker plot of loss per trained model
FIGNAME = "sampling_rate_box_whisker"
FIGSIZE = (11*sf, 5*sf)

show_outliers = False
markers = ['^', 'v', '<', '>']
colors = ['green', 'orange', 'blue']
offset = 0.2

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

for dt_idx, dt in enumerate(dt_list):
    datdir = DT_TO_DATDIR[dt]
    model_list, epoch_list = get_models_sorted(dt)

    num_models = len(model_list)
    total_width = 0.9
    hspace_factor = 0.1
    boxwidth = total_width / (num_models * (1 + hspace_factor) - hspace_factor)
    offset = boxwidth + hspace_factor * boxwidth
    for i, m in enumerate(model_list):
        loss_data_dir = f"{BASEMODELDIR}/{m}/eval"
        filelist = [
            f.removesuffix('.npy') 
            for f in os.listdir(loss_data_dir) 
            if f.endswith(".npy")
        ]
        filelist = [s.split('_') for s in filelist]
        dt0_values = [float(v[3]) for v in filelist]
        dt0 = np.min(dt0_values)
        # print(dt, i, dt0)
        losses = np.load(f"{loss_data_dir}/losses_test_dt_{dt0}.npy")
        conditions = np.load(f"{loss_data_dir}/conditions_test_dt_{dt0}.npy")
        nconds, ndata_per_cond, nresamps, nreps = losses.shape
        assert conditions.shape == (nconds * ndata_per_cond * nresamps, 2, 4)
        assert nresamps == 1, "nresamps should be 1"
        conditions = conditions.reshape((nresamps, nconds, ndata_per_cond, 2, 4))
        conditions = conditions.transpose(1, 2, 0, 3, 4)
        losses = losses.squeeze(2)
        conditions = conditions.squeeze(2)
        assert losses.shape == (nconds, ndata_per_cond, nreps), "bad loss shape"
        assert conditions.shape == (nconds, ndata_per_cond, 2, 4), \
            f"bad cond shape. Expected: {(nconds, ndata_per_cond, 2, 4)}. Got {conditions.shape}"
        
        losses = np.mean(losses, axis=-1)
        med_loss = np.median(losses)
        
        xpos = 1 + dt_idx + i*offset - num_models/2*offset
        if num_models % 2 == 0:
            xpos += offset / 2

        bplot = ax.boxplot(
            losses.flatten(), 0, '.' if show_outliers else '',
            patch_artist=True,
            positions=[xpos],
            widths=[boxwidth],
            medianprops={'color': 'r'},
            flierprops={'color': 'k', 'alpha': 0.5} if show_outliers else None,
        )
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[dt_idx])


# ax.text(
#     0.01, 0.99, f"outliers included" if show_outliers else "outliers excluded", 
#     fontsize=6,
#     ha='left', va='top', 
#     transform=ax.transAxes
# )
ax.set_xlim(ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset)
ax.set_xticks(range(1, len(dt_list) + 1), dt_list)
ax.set_xlabel("sampling interval $\Delta T$")
ax.set_ylabel("error")
ax.set_title("Generalization error")

plt.savefig(f"{OUTDIR}/{FIGNAME}.pdf")


##############################################################################
##############################################################################
####  Loss vs closeness metric
FIGNAME = "loss_vs_nnd"
FIGSIZE = (8*sf, 5*sf)

markers = ['^', 'v', '<', '>']
colors = ['green', 'orange', 'blue']
offset = 0.2
threshold_to_plot = THRESHOLD

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

for dt_idx, dt in enumerate(dt_list):
    datdir = DT_TO_DATDIR[dt]
    model_list, epoch_list = get_models_sorted(dt)

    datdir_train = f"{datdir}/training"
    datdir_valid = f"{datdir}/validation"
    datdir_test = f"{datdir}/testing"
    nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
    nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
    nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)
    _, _, _, _, _, test_dset = get_dataloaders(
        datdir_train, datdir_valid, nsims_train, nsims_valid,
        shuffle_train=False,
        shuffle_valid=False,
        return_datasets=True,
        include_test_data=True,
        datdir_test=datdir_test, nsims_test=nsims_test, shuffle_test=False,
        batch_size_test=1,
        seed=rng.integers(2**32)
    )

    num_models = len(model_list)
    for i, m in enumerate(model_list):
        loss_data_dir = f"{BASEMODELDIR}/{m}/eval"
        filelist = [
            f.removesuffix('.npy') 
            for f in os.listdir(loss_data_dir) 
            if f.endswith(".npy")
        ]
        filelist = [s.split('_') for s in filelist]
        dt0_values = [float(v[3]) for v in filelist]
        dt0 = np.min(dt0_values)
        losses = np.load(f"{loss_data_dir}/losses_test_dt_{dt0}.npy")
        conditions = np.load(f"{loss_data_dir}/conditions_test_dt_{dt0}.npy")
        nconds, ndata_per_cond, nresamps, nreps = losses.shape
        assert conditions.shape == (nconds * ndata_per_cond * nresamps, 2, 4)
        assert nresamps == 1, "nresamps should be 1"
        conditions = conditions.reshape((nresamps, nconds, ndata_per_cond, 2, 4))
        conditions = conditions.transpose(1, 2, 0, 3, 4)
        losses = losses.squeeze(2)
        conditions = conditions.squeeze(2)
        assert losses.shape == (nconds, ndata_per_cond, nreps), "bad loss shape"
        assert conditions.shape == (nconds, ndata_per_cond, 2, 4), \
            f"bad cond shape. Expected: {(nconds, ndata_per_cond, 2, 4)}. Got {conditions.shape}"

        losses = np.mean(losses, axis=-1)  # average over reps

        dist_scores = np.zeros(losses.shape)
        for condidx in range(nconds):
            for tpidx in range(ndata_per_cond):
                data_idx = condidx * ndata_per_cond + tpidx
                td1 = test_dset[data_idx]
                td2 = test_dset[data_idx]
                assert np.all(td1[0][1] == td2[0][1]) and np.all(td1[1] == td2[1])
                (t0, x0, t1, sp), x1 = test_dset[data_idx]
                assert np.allclose(sp, conditions[condidx, tpidx])
                nbrs0 = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(x0)
                dists0, idxs0 = nbrs0.kneighbors(x0)
                nbrs1 = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(x1)
                dists1, idxs1 = nbrs1.kneighbors(x1)
                # avg_dist0 = np.percentile(np.mean(dists0[:,1:], axis=-1), 95)
                # avg_dist1 = np.percentile(np.mean(dists1[:,1:], axis=-1), 95)
                avg_dist0 = np.mean(dists0[:,1:])
                avg_dist1 = np.mean(dists1[:,1:])
                dist_score = (avg_dist0 + avg_dist1) / 2
                dist_scores[condidx, tpidx] = dist_score
                if losses[condidx,tpidx] >= threshold_to_plot:
                    fig2, ax2 = plt.subplots(1, 1, figsize=None)

                    ax2.plot(
                        x0[:,0], x0[:,1], '.',
                        color='k',
                        alpha=0.5,
                    )
                    ax2.plot(
                        x1[:,0], x1[:,1], '.',
                        color='r',
                        alpha=0.5,
                    )
                    ax2.text(
                        0.5, 0.99, f"loss: {losses[condidx,tpidx]}", 
                        fontsize=6,
                        ha='center', va='top', 
                        transform=ax2.transAxes 
                    )
                    os.makedirs(f"{OUTDIR}/datapoints", exist_ok=True)
                    plt.savefig(f"{OUTDIR}/datapoints/dt{dt}_{m}_data_c{condidx}_t{tpidx}.pdf")
                    plt.close()

        # losses = np.log10(losses - losses.min() + 1)
        ax.plot(
            np.log10(dist_scores.flatten()), losses.flatten(), 
            marker='.', 
            markersize=3, 
            linestyle="None",
            color=colors[dt_idx], 
            alpha=0.5,
            zorder=len(dt_list) - dt_idx,
            label=f"$\Delta T={dt}$" if i == 0 else None
        )

ax.text(
    0.99, 0.99, f"outliers included", 
    fontsize=6,
    ha='right', va='top', 
    transform=ax.transAxes
)

ax.legend(loc='upper left')
ax.set_xlabel("log avg dist to 10-nearest neighbors")
ax.set_ylabel("loss")
ax.set_title("Evaluation loss vs local closeness")

plt.figure(fig)
plt.savefig(f"{OUTDIR}/{FIGNAME}.pdf")
plt.close()


##############################################################################
##############################################################################
####  Landscape Plots
FIGSIZE = (2.5*sf, 2.5*sf)
SIG_TO_PLOT = [-0.5, 1]

for dt_idx, dt in enumerate(dt_list):
    datdir = DT_TO_DATDIR[dt]
    model_list, epoch_list = get_models_sorted(dt)
    num_models = len(model_list)
    for i, m in enumerate(model_list):
        modeldir = f"{BASEMODELDIR}/{m}"
        loss_data_dir = f"{modeldir}/eval"
        
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

        filelist = [
            f.removesuffix('.npy') 
            for f in os.listdir(loss_data_dir) 
            if f.endswith(".npy")
        ]
        filelist = [s.split('_') for s in filelist]
        dt0_values = [float(v[3]) for v in filelist]
        dt0 = np.min(dt0_values)
        losses = np.load(f"{loss_data_dir}/losses_test_dt_{dt0}.npy")
        conditions = np.load(f"{loss_data_dir}/conditions_test_dt_{dt0}.npy")
        nconds, ndata_per_cond, nresamps, nreps = losses.shape
        assert conditions.shape == (nconds * ndata_per_cond * nresamps, 2, 4)
        assert nresamps == 1, "nresamps should be 1"
        conditions = conditions.reshape((nresamps, nconds, ndata_per_cond, 2, 4))
        conditions = conditions.transpose(1, 2, 0, 3, 4)
        losses = losses.squeeze(2)
        conditions = conditions.squeeze(2)
        assert losses.shape == (nconds, ndata_per_cond, nreps), "bad loss shape"
        assert conditions.shape == (nconds, ndata_per_cond, 2, 4), \
            f"bad cond shape. Expected: {(nconds, ndata_per_cond, 2, 4)}. Got {conditions.shape}"
                
        med_loss = np.median(losses)
        mean_loss = np.mean(losses)
        
        model, _, epoch_loaded, _, _ = load_model_from_directory(
            modeldir
        )

        model.plot_phi(
            signal=SIG_TO_PLOT,
            r=2,
            res=50,
            show=True,
            include_cbar=False,
            cbar_title="",
            title="",
            xlabel=None, ylabel=None,
            xticks=False, yticks=False,
            ax=ax,
        )
        # ax.text(
        #     0.99, 0.01, f"epoch: {epoch_loaded}\n$\mathtt{{dt}}$: {model.get_dt0()}", 
        #     fontsize=6,
        #     ha='right', va='bottom', 
        #     transform=ax.transAxes
        # )

        plt.savefig(
            f"{OUTDIR}/model_dt_{dt}_{i}.pdf", 
            bbox_inches='tight', pad_inches = 0
        )
        plt.close()


##############################################################################
##############################################################################
##  Training/testing dataset scatter plots
        
FIGSIZE = (2.5*sf, 2.5*sf)
NIDXS_HIGHLIGHT = 0
highlight_colors = ['k', 'r']

for dt_idx, dt in enumerate(dt_list):
    datdir = DT_TO_DATDIR[dt]
    model_list, epoch_list = get_models_sorted(dt)

    datdir_train = f"{datdir}/training"
    datdir_valid = f"{datdir}/validation"
    datdir_test = f"{datdir}/testing"
    nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
    nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
    nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)
    _, _, _, train_dset, _, test_dset = get_dataloaders(
        datdir_train, datdir_valid, nsims_train, nsims_valid,
        shuffle_train=False,
        shuffle_valid=False,
        return_datasets=True,
        include_test_data=True,
        datdir_test=datdir_test, nsims_test=nsims_test, shuffle_test=False,
        batch_size_test=1,
        seed=rng.integers(2**32)
    )

    for dset, dset_name in zip([train_dset, test_dset], ['train', 'test']):

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

        ndatapoints = dset.get_baselength()
        ndata_per_cond = 20 / dt
        nsnapshots = ndata_per_cond + 1
        nconds = ndatapoints / ndata_per_cond
        ncells = len(x1)

        print(VERSION, dt, dset_name)
        print("  ndatapoints:", ndatapoints)
        print("  ndata_per_cond:", ndata_per_cond)
        print("  nsnapshots:", nsnapshots)
        print("  nconds:", nconds)
        print("  ncells:", ncells)
        
        for d in dset:
            x0 = d[0][1]
            x1 = d[1]
            ax.plot(
                x0[:,0], x0[:,1], '.', 
                color=colors[dt_idx],
                markersize=1, 
                alpha=0.5, 
                rasterized=True
            )
            ax.plot(
                x1[:,0], x1[:,1], '.', 
                color=colors[dt_idx],
                markersize=1, 
                alpha=0.5, 
                rasterized=True
            )

        dataidxs = np.sort(
            rng.choice(len(dset), size=NIDXS_HIGHLIGHT, replace=False)
        )
        for j, dataidx in enumerate(dataidxs):
            x0 = dset[dataidx][0][1]
            x1 = dset[dataidx][1]
            l, = ax.plot(
                x0[:,0], x0[:,1], '.', markersize=1, alpha=1,
                label=f"obs {dataidx} (n={len(x0)})",
                rasterized=False,
                color=highlight_colors[j],
            )
            ax.plot(
                x1[:,0], x1[:,1], '.', markersize=1, alpha=1,
                color=l.get_color(),
                rasterized=False,
            )

        # string = f"$N_{{\\text{{exps}}}}$: {nconds}"
        # string += f"\n$|\mathcal{{D}}_i|$: {ndata_per_cond}"
        # ax.text(
        #     0.5, 0.99, string, 
        #     fontsize=6,
        #     ha='center', va='top', 
        #     transform=ax.transAxes
        # )
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$y$")
        ax.set_title("");

        train_ax_xlims = ax.get_xlim()
        train_ax_ylims = ax.get_ylim()

        plt.savefig(f"{OUTDIR}/data_dt_{dt}_{dset_name}.pdf", bbox_inches='tight')
        plt.close()


##############################################################################
##############################################################################
##  Training/testing dataset density plots
        
FIGSIZE = (2.5*sf, 2.5*sf)
NIDXS_HIGHLIGHT = 0
highlight_colors = ['k', 'r']

xrange = train_ax_xlims
yrange = train_ax_ylims
dx = dy = 0.05

for dt_idx, dt in enumerate(dt_list):
    datdir = DT_TO_DATDIR[dt]
    model_list = DT_TO_MODELS[dt]

    datdir_train = f"{datdir}/training"
    datdir_valid = f"{datdir}/validation"
    datdir_test = f"{datdir}/testing"
    nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
    nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
    nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)
    _, _, _, train_dset, _, test_dset = get_dataloaders(
        datdir_train, datdir_valid, nsims_train, nsims_valid,
        shuffle_train=False,
        shuffle_valid=False,
        return_datasets=True,
        include_test_data=True,
        datdir_test=datdir_test, nsims_test=nsims_test, shuffle_test=False,
        batch_size_test=1,
        seed=rng.integers(2**32)
    )

    for dset, dset_name in zip([train_dset, test_dset], ['train', 'test']):

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

        ndatapoints = dset.get_baselength()
        ndata_per_cond = 20 / dt
        nsnapshots = ndata_per_cond + 1
        nconds = ndatapoints / ndata_per_cond
        ncells = len(x1)

        print(VERSION, dt, dset_name)
        print("  ndatapoints:", ndatapoints)
        print("  ndata_per_cond:", ndata_per_cond)
        print("  nsnapshots:", nsnapshots)
        print("  nconds:", nconds)
        print("  ncells:", ncells)
        
        xs_all = dset.get_all_cells().reshape([-1, 2])
        edges_x = np.linspace(*xrange, 1 + round((xrange[1] - xrange[0])/dx) )
        edges_y = np.linspace(*yrange, 1 + round((yrange[1] - yrange[0])/dy) )
        hist2d = get_hist2d(xs_all, edges_x, edges_y)
        ax.imshow(
            hist2d, origin='lower', aspect='auto', 
            extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]],
            cmap='viridis',
            norm='log',
            interpolation='none',
        )

        plt.savefig(f"{OUTDIR}/density_dt_{dt}_{dset_name}.pdf", bbox_inches='tight')
        plt.close()

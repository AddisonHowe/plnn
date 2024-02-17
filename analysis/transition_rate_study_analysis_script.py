"""Analysis script for models trained for the transition rate study.

"""

import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

import jax.numpy as jnp
import jax.random as jrandom

from plnn.models import DeepPhiPLNN
from plnn.dataset import get_dataloaders
from plnn.data_generation.signals import get_sigmoid_function

from notebooks.helpers import load_model_directory
from notebooks.plot_landscapes import plot_landscape, func_phi1, func_phi2

from notebooks.plotting import plot_loss_history, plot_learning_rate_history
from notebooks.plotting import plot_sigma_history, plot_phi_inferred_vs_true


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--trsdir', type=str, 
                        default="data/transition_rate_study_results")
    parser.add_argument('--datdir', type=str, 
                        default="transition_rate_study_model_training_kl1")
    parser.add_argument('--outdir', type=str, 
                        default="out/transition_rate_study_analysis")
    parser.add_argument('--training_datdir', type=str,
                        default="data/transition_rate_studies")
    parser.add_argument('--load_index', type=int, default=-1,
                        help="Index of model to load. If -1, load best.")
    return parser.parse_args(args)


def _parse_model_training_id(model_training_id, trial_name, 
                           model_prefix="model_"):
    model_name = model_prefix + trial_name
    suffix = model_training_id.removeprefix(f"{model_name}_r")
    x = suffix.split("_", 1)
    ridx = int(x[0])
    timestamp = "" if len(x) == 1 else x[1]
    return model_name, ridx, timestamp


def run_analysis(
        trial_name,
        ridx,
        modeldir,
        model_name,
        datdir_train,
        datdir_valid,
        load_index,
        outdir,
):
    logr_fpath = f"{datdir_valid}/logr.txt"
    sigma_true_fpath = f"{datdir_valid}/../sigma"
    landscape_true_fpath = f"{datdir_valid}/../landscape_name"
    nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
    nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
    landscape_true = np.genfromtxt(landscape_true_fpath, dtype=str)
    logr = np.genfromtxt(logr_fpath, dtype=int)
    sigma_true = np.genfromtxt(sigma_true_fpath, dtype=float)

    train_dloader, valid_dloader, train_dset, valid_dset = get_dataloaders(
        datdir_train=datdir_train,
        datdir_valid=datdir_valid,
        nsims_train=nsims_train,
        nsims_valid=nsims_valid,
        shuffle_train=False,
        shuffle_valid=False,
        ndims=2,
        return_datasets=True,
    )

    optidx, logged_args, run_dict = load_model_directory(
        modeldir, model_name, subdir="states"
    )

    loss_train = run_dict['loss_hist_train']
    loss_valid = run_dict['loss_hist_valid']
    sigma_hist = run_dict['sigma_hist']
    lr_hist    = run_dict['learning_rate_hist']
    loss_method = logged_args['loss']
    optimizer   = logged_args['optimizer']

    if load_index == -1:
        load_index = optidx
    fpath_to_load = f"{modeldir}/states/{model_name}_{1 + load_index}.pth"
    model, hyperparams = DeepPhiPLNN.load(fpath_to_load, dtype=jnp.float64)
    
    plot_loss_history(
        loss_train, loss_valid,
        startidx=0, 
        log=True, 
        title=f"Loss History ({loss_method}, {optimizer})",
        saveas=f"{outdir}/loss_history.png"
    )
    plt.close()

    plot_learning_rate_history(
        lr_hist, log=False, 
        saveas=f"{outdir}/lr_history.png"
    )
    plt.close()

    plot_sigma_history(
        sigma_hist, log=False, 
        sigma_true=sigma_true,
        saveas=f"{outdir}/sigma_history.png"
    )
    plt.close()

    plot_phi_inferred_vs_true(
        model, [0,0], landscape_true, 
        ax1_title="Inferred",
        ax2_title="True",
        axes=None,
        figsize=(12,5),
        plot_radius=4,
        plot_res=200,
        lognormalize=True,
        saveas=f"{outdir}/landscape_true_vs_inferred.pdf",
    )
    plt.close()


def main(args):
    datdir = f"{args.trsdir}/{args.datdir}"
    outdir = args.outdir
    training_datdir = args.training_datdir
    load_index = args.load_index

    os.makedirs(outdir, exist_ok=True)

    trial_list = os.listdir(datdir)

    for trial_name in trial_list:
        trial_dir = f"{datdir}/{trial_name}"
        model_training_id_list = os.listdir(trial_dir)
        for model_training_id in model_training_id_list:
            model_name, ridx, tstamp = _parse_model_training_id(
                model_training_id, trial_name)
            modeldir = f"{trial_dir}/{model_training_id}"
            trndirbase = f"{training_datdir}/{trial_name}"
            datdir_train = f"{trndirbase}/{trial_name}_training/r{ridx}"
            datdir_valid = f"{trndirbase}/{trial_name}_validation/r{ridx}"

            imgdir = f"{outdir}/{model_training_id}"
            os.makedirs(imgdir, exist_ok=True)

            run_analysis(
                trial_name,
                ridx,
                modeldir,
                model_name,
                datdir_train,
                datdir_valid,
                load_index=load_index,
                outdir=imgdir,
            )


            



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

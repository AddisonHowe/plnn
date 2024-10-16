#!/bin/bash

# Process model training arg files located in the directory `datdir` and
# create a markdown table summarizing the parameters.

datdir=data/model_training_args

fnames=(
    run_phi1_1a_v_kl1
    run_phi1_1a_v_kl2
    run_phi1_1a_v_kl3
    run_phi1_1a_v_kl4
    run_phi1_1a_v_kl5
    run_phi1_1a_v_mmd1
    run_phi1_1a_v_mmd2
    run_phi1_1a_v_mmd3
    run_phi1_1a_v_mmd4
    run_phi1_1a_v_mmd5
    run_phi1_1a_v_mmd6
    run_phi1_1a_v_mmd1_fix_noise_small
    run_phi1_1a_v_mmd1_fix_noise_large
    run_phi1_1a_v_mmd1_smallnet
    run_phi1_1a_v_mmd1_largenet
    run_phi1_1a_v_mmd1_presample
    run_phi2_1a_v_mmd1
    run_phi2_1a_v_mmd2
    run_phi2_1a_v_mmd4
    run_phi2_1a_v_mmd5
    run_phiq_1a_v_mmd1
)

cols=(
    argfile
    "training data"
    nepochs
    patience
    "batch size"
    "phi layers"
    ncells
    sigma
    loss
    solver
    dt0
    "dt scheduling"
    "learning rate"
    optimizer
)

header="|"
seprow="|"
for col in "${cols[@]}"; do
    header=$header" "$col" |"
    seprow=$seprow" --- |"
done

foo () {
    fpath=$1
    awk -v fpath="$fpath" -F "\t+" '
    NR>0 {
        if (substr($1,1,1)!="#") dict[$1]=$2;
    } END {
        name=dict["name"];
        num_epochs=dict["num_epochs"];
        batch_size=dict["batch_size"];
        ncells=dict["ncells"];
        sigma=dict["sigma"];
        fix_noise=dict["fix_noise"];
        if (fix_noise == "True")
            sigma_args="(fixed)"
        else
            sigma_args=""
        patience=dict["patience"];
        dt=dict["dt"];
        dt_schedule=dict["dt_schedule"];
        dt_schedule_bounds=dict["dt_schedule_bounds"];
        dt_schedule_scales=dict["dt_schedule_scales"];
        phi_layers=dict["phi_hidden_dims"];
        traindir=dict["training_data"];
        validdir=dict["validation_data"];
        loss=dict["loss"];
        solver=dict["solver"];
        lr_schedule=dict["lr_schedule"];
        if (lr_schedule == "exponential_decay") 
            lr_args="<br>("dict["learning_rate"]", "dict["final_learning_rate"]", "dict["nepochs_warmup"]")"
        else if (lr_schedule == "warmup_cosine_decay") 
            lr_args="<br>("dict["learning_rate"]", "dict["peak_learning_rate"]", "dict["final_learning_rate"]", "dict["nepochs_warmup"]")"
        if (dt_schedule == "stepped")
            dt_schedule_args="<br>bounds: ["dt_schedule_bounds"]<br>scales: ["dt_schedule_scales"]"
        optimizer=dict["optimizer"]
        if (optimizer == "rms")
            optim_args="<br>m="dict["momentum"]"<br>decay="dict["weight_decay"]"<br>clip="dict["clip"]
        # Build columns
        print "| " "["name"]("fpath")" " | " "[training data]("traindir")<br>[validation data]("validdir")" " | " num_epochs " | " patience " | " batch_size " | " phi_layers " | " ncells " | " sigma" "sigma_args " | " loss " | " solver " | " dt " | " dt_schedule" "dt_schedule_args " | " lr_schedule" "lr_args" | " optimizer" "optim_args " |"
    }
    ' < "$fpath"
}

echo $header
echo $seprow
for fname in ${fnames[@]}; do
    fpath=$datdir/$fname.tsv 
    foo $fpath
done

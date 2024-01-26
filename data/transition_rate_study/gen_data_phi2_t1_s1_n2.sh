#!/bin/sh

index=$1

runid=phi2_t1_s1_n2

nsims_train=2
nsims_valid=2

sigma="1e-2"
tfin=10
dt="1e-3"
dt_save=1.0

logr_range="-2 2"
num_rs=6
tcrit=5.0

landscape_name=phi2
x0="-1 0"
ncells=500
burnin=0.5

# Animation options
duration=10
animation_dt=0.1
sims_to_animate="0"

python plnn/data_generation/transition_rate_simulations.py \
    --outdir data/transition_rate_study/${runid}_training \
    --index ${index} \
    --landscape_name ${landscape_name} \
    --sigma ${sigma} \
    --tcrit ${tcrit} \
    --logr_range ${logr_range} \
    --num_rs ${num_rs} \
    --nsims ${nsims_train} \
    --ncells ${ncells} \
    --x0 ${x0} \
    --tfin ${tfin} \
    --dt ${dt} \
    --dt_save ${dt_save} \
    --burnin ${burnin} \
    --animate \
    --duration ${duration} \
    --animation_dt ${animation_dt} \
    --sims_to_animate ${sims_to_animate} \
    --seed 999992112 \

python plnn/data_generation/transition_rate_simulations.py \
    --outdir data/transition_rate_study/${runid}_validation \
    --index ${index} \
    --landscape_name ${landscape_name} \
    --sigma ${sigma} \
    --tcrit ${tcrit} \
    --logr_range ${logr_range} \
    --num_rs ${num_rs} \
    --nsims ${nsims_valid} \
    --ncells ${ncells} \
    --x0 ${x0} \
    --tfin ${tfin} \
    --dt ${dt} \
    --dt_save ${dt_save} \
    --burnin ${burnin} \
    --animate \
    --duration ${duration} \
    --animation_dt ${animation_dt} \
    --sims_to_animate ${sims_to_animate} \
    --seed 888882112 \
    
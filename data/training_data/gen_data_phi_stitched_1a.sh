#!/bin/bash

runname="data_phi_stitched_1a"

landscape="phi_stitched"

nsims_train=100
nsims_valid=30

seed_train=24982398
seed_valid=22935534

sigma=0.085
tfin=72.0
dt=0.001
dt_save=12.0
ncells=500
burnin=0.05
nsignals=2
noise_schedule=constant
x0="0.0 -0.5"

animation_dt=1.0
sims_to_animate="0 1 2 3"
animation_duration=10

echo Logging information to logs/${runname}.o
echo "Generating training data..."

python plnn/data_generation/gen_stitched_landscape_data.py \
    -o data/training_data/${runname}/training \
    --nsims $nsims_train \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_train \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
> logs/${runname}.o 

echo "Generating validation data..."

python plnn/data_generation/gen_stitched_landscape_data.py \
    -o data/training_data/${runname}/validation \
    --nsims $nsims_valid \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_valid \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
>> logs/${runname}.o 

echo "Done!"

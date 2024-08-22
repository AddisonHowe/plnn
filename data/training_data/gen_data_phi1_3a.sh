#!/bin/bash

runname="data_phi1_3a"

landscape="phi1"
s10_range="-0.5 0.5"
s20_range="0.5 1.5"
s11_range="-1 1"
s21_range="-0.5 0.5"
logr1_range="3 4"
logr2_range="3 4"
logr1_range="3 4"
logr2_range="3 4"
tcrit_buffer0=0.1
tcrit_buffer1=0.85

nsims_train=50
nsims_valid=10
nsims_test=10

seed_train=76678345
seed_valid=734765
seed_test=87326534

sigma=0.1
tfin=20
dt=0.001
dt_save=5.0
ncells=50
burnin=0.1
nsignals=2
signal_schedule=sigmoid
param_func=identity
noise_schedule=constant
x0="0 -0.5"

animation_dt=0.1
sims_to_animate="0 1 2"
animation_duration=10

echo Logging information to logs/${runname}.o
echo "Generating training data..."

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname}/training \
    --nsims $nsims_train \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
    --tcrit_buffer0 ${tcrit_buffer0} \
    --tcrit_buffer1 ${tcrit_buffer1} \
    --param_func $param_func \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_train \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
> logs/${runname}.o 

echo "Generating validation data..."

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname}/validation \
    --nsims $nsims_valid \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
    --tcrit_buffer0 ${tcrit_buffer0} \
    --tcrit_buffer1 ${tcrit_buffer1} \
    --param_func $param_func \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_valid \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
>> logs/${runname}.o 

echo "Generating testing data..."

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname}/testing \
    --nsims $nsims_test \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
    --tcrit_buffer0 ${tcrit_buffer0} \
    --tcrit_buffer1 ${tcrit_buffer1} \
    --param_func $param_func \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_test \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
>> logs/${runname}.o 

echo "Done!"

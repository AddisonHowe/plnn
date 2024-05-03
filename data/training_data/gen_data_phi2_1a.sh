#!/bin/bash

runname="data_phi2_1a"

landscape="phi2"
s10_range="-1.5 -1.0"
s20_range="-0.75 0.75"
s11_range="-1.5 -1.0"
s21_range="-0.75 0.75"
logr1_range="-3 2"
logr2_range="-3 2"

nsims_train=100
nsims_valid=30
nsims_test=30

seed_train=1248778235
seed_valid=2395762349
seed_test=98391248723114

sigma=0.3
tfin=100
dt=0.001
dt_save=10.0
ncells=500
burnin=0.05
burnin_signal="-0.25 0.00"
nsignals=2
signal_schedule=sigmoid
param_func=identity
noise_schedule=constant
x0="-1.0 0.0"

animation_dt=1.0
sims_to_animate="0 1 2 3"
animation_duration=10

echo Logging information to logs/${runname}.o
echo "Generating training data..."

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname}/training \
    --nsims $nsims_train \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --burnin_signal $burnin_signal \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
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
    --burnin_signal $burnin_signal \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
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
    --burnin_signal $burnin_signal \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
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

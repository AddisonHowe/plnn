#!/bin/bash

runname="data_phi2_3a_training"

landscape="phi2"
sigma=0.01
s10_range="-2 -0.5"
s20_range="-0.5 0.5"
s11_range="-2 -0.5"
s21_range="-0.5 0.5"

echo Logging information to logs/${runname}.o

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname} \
    --nsims 1000 \
    -t 10 --dt 0.001 --dt_save 2.0 --ncells 500 --burnin 0.1 \
    --landscape_name ${landscape} \
    --nsignals 2 \
    --signal_schedule binary \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --param_func identity \
    --noise_schedule constant --noise_args ${sigma} \
    --x0 "-1.0" 0.0 \
    --seed 1313123123 \
    --animate \
    --duration 10 \
    --animation_dt 0.1 \
    --sims_to_animate 0 1 2 3 4 5 6 7 8 9 \
> logs/${runname}.o 

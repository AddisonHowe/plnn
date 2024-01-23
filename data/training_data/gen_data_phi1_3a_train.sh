#!/bin/bash

runname="data_phi1_3a_training"

landscape="phi1"
sigma=0.01
s10_range="-1 1"
s20_range=" 1 2"
s11_range="-1.5 1.5"
s21_range="-1 0.25"

echo Logging information to logs/${runname}.o

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname} \
    --nsims 1000 \
    -t 10 --dt 0.001 --dt_save 2.0 --ncells 500 --burnin 100 \
    --landscape_name ${landscape} \
    --nsignals 2 \
    --signal_schedule binary \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --param_func identity \
    --noise_schedule constant --noise_args ${sigma} \
    --x0 0 "-0.5" \
    --seed 13245 \
    --animate \
    --duration 10 \
    --animation_dt 0.1 \
    --sims_to_animate 0 1 2 3 4 5 6 7 8 9 \
> logs/${runname}.o 

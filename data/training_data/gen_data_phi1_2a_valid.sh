#!/bin/bash

runname="data_phi1_2a_validation"

landscape="phi1"
sigma=0.01
s10_range="-1 1"
s20_range="-1 1"
s11_range="-1 1"
s21_range="-1 1"

echo Logging information to logs/${runname}.o

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname} \
    --nsims 30 \
    --tfin 100 --dt 0.001 --dt_save 1.0 --ncells 500 --burnin 0.1 \
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
    --seed 54632 \
    --animate \
    --duration 10 \
    --animation_dt 1.0 \
    --sims_to_animate 0 1 \
> logs/${runname}.o 

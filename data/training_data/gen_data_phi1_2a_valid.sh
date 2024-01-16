#!/bin/bash

runname="data_phi1_2a_validation"

landscape="phi1"
sigma=0.01
p10_range="-1 1"
p20_range="-1 1"
p11_range="-1 1"
p21_range="-1 1"

echo Logging information to logs/${runname}.o

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname} \
    --nsims 30 \
    -t 100 --dt 0.001 --dt_save 1.0 --ncells 500 --burnin 100 \
    --landscape_name ${landscape} \
    --param_schedule binary \
    --noise_schedule constant --noise_args ${sigma} \
    --x0 0 "-0.5" \
    --p10_range ${p10_range} \
    --p20_range ${p20_range} \
    --p11_range ${p11_range} \
    --p21_range ${p21_range} \
    --seed 54632 \
> logs/${runname}.o 

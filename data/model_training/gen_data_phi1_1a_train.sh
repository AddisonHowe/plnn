#!/bin/bash

runname="data_phi1_1a_training"

landscape="phi1"
sigma=0.001
p10_range="-1 1"
p20_range="-1 1"
p11_range="-1 1"
p21_range="-1 1"

echo Logging information to logs/${runname}.o

python plnn/data_generation/generate_data.py -o data/${runname} --nsims 100 \
    -t 10 --dt 0.001 --dt_save 0.1 --ncells 100 --burnin 50 \
    --landscape_name ${landscape} \
    --param_schedule binary \
    --noise_schedule constant --noise_args ${sigma} \
    --x0 0 "-0.5" \
    --p10_range ${p10_range} \
    --p20_range ${p20_range} \
    --p11_range ${p11_range} \
    --p21_range ${p21_range} \
    --seed 13245 \
> logs/${runname}.o 

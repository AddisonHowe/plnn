#!/bin/bash

runname="data_phi2_3a_training"

landscape="phi2"
sigma=0.01
p10_range="-2 -0.5"
p20_range="-0.5 0.5"
p11_range="-2 -0.5"
p21_range="-0.5 0.5"

echo Logging information to logs/${runname}.o

python plnn/data_generation/generate_data.py \
    -o data/training_data/${runname} \
    --nsims 1000 \
    -t 10 --dt 0.001 --dt_save 2.0 --ncells 500 --burnin 100 \
    --landscape_name ${landscape} \
    --param_schedule binary \
    --noise_schedule constant --noise_args ${sigma} \
    --x0 "-1.0" 0.0 \
    --p10_range ${p10_range} \
    --p20_range ${p20_range} \
    --p11_range ${p11_range} \
    --p21_range ${p21_range} \
    --seed 1313123123 \
> logs/${runname}.o 

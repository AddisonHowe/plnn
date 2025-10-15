#!/bin/bash

p10_range="-1 1"
p20_range="-1 1"
p11_range="-1 1"
p21_range="-1 1"

PYTESTDATDIR=$(python -c "from tests.conftest import DATDIR; print(DATDIR)")

python plnn/data_generation/generate_data.py \
    -o $PYTESTDATDIR/test_main_data/training_data \
    --nsims 20 \
    -t 10 --dt 0.01 --dt_save 2.0 --ncells 50 --burnin 50 \
    --landscape_name phi1 \
    --param_schedule binary \
    --noise_schedule constant --noise_args 0.001 \
    --x0 0 "-0.5" \
    --p10_range ${p10_range} \
    --p20_range ${p20_range} \
    --p11_range ${p11_range} \
    --p21_range ${p21_range} \
    --seed 123

python plnn/data_generation/generate_data.py \
    -o $PYTESTDATDIR/test_main_data/validation_data \
    --nsims 10 \
    -t 10 --dt 0.01 --dt_save 2.0 --ncells 50 --burnin 50 \
    --landscape_name phi1 \
    --param_schedule binary \
    --noise_schedule constant --noise_args 0.001 \
    --x0 0 "-0.5" \
    --p10_range ${p10_range} \
    --p20_range ${p20_range} \
    --p11_range ${p11_range} \
    --p21_range ${p21_range} \
    --seed 321
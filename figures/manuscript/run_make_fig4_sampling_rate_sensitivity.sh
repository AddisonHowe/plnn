#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig4_sampling_rate_sensitivity.py -v phi1_2
python figures/manuscript/make_fig4_sampling_rate_sensitivity.py -v phi1_3
python figures/manuscript/make_fig4_sampling_rate_sensitivity.py -v phi1_4

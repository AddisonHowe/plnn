#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig3_synthetic_training.py

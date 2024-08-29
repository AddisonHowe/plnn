#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig1_landscape_models.py

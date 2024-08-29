#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig5_dimred_schematic.py

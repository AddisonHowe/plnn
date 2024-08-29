#!/bin/bash

sh figures/manuscript/pull_fig6_facs_training.sh

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig6_facs_training.py


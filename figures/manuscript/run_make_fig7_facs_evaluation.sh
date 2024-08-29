#!/bin/bash

sh figures/manuscript/pull_fig7_facs_evaluation.sh

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig7_facs_evaluation.py
python figures/manuscript/make_fig7_facs_evaluation_2.py
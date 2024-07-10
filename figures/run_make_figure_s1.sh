#!/bin/bash

python figures/make_figure_s1.py -i "model_phi1_1a_v_mmd1_20240522_185135" --truesigma 0.1
python figures/make_figure_s1.py -i "model_phi2_1a_v_mmd1_20240523_093008" --truesigma 0.3 --no-logloss --startidx 50

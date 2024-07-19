#!/bin/bash

fpath=scripting/arglist_nb_eval_model_facs.tsv

awk '!/^#/ { count++ } END { print count - 1 }' $fpath

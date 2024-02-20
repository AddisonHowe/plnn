#!/bin/bash

results_dir=$1

for studydir in data/transition_rate_study_results/$results_dir/*; do 
    # echo $studydir
    outdir=out/bifs/$results_dir/$(basename $studydir)
    echo $outdir
    mkdir -p $outdir
    python analysis/transition_rate_study_analysis_script.py \
        --trsdir data/transition_rate_study_results \
        --datdir $results_dir \
        --studydir $(basename $studydir) \
        --outdir $outdir \
        --training_datdir data/transition_rate_studies
done

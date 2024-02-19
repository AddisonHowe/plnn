#!/bin/bash

results_dir=$1

for studydir in data/transition_rate_study_results/$results_dir/*; do 
    for modeldir in $studydir/*; do
        echo $modeldir;
        outdir=tmpbifs/$results_dir/$(basename $studydir)/$(basename $modeldir)
        mkdir -p $outdir
        for ((i = 0; i < 1; i++)); do
            python cont/plnn_bifurcations.py \
            --modeldir $modeldir \
            -v 0 -n 400 --progress_bar \
            --plot_first_steps --plot_failed_to_converge_points \
            --saveas $outdir/bif_diagram$i.png \
            --savedata --outdir $outdir/bifcurve_data
        done
    done        
done

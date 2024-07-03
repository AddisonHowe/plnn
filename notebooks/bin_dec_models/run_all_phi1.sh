#!/bin/bash

rundir=notebooks/bin_dec_models
argfile=$rundir/phi1_arglist.tsv

jupyter nbconvert --RegexRemovePreprocessor.patterns="^%" \
    --to script $rundir/nb_eval_model_phi1.ipynb \
    --output tmp_script

eval "$(conda shell.bash hook)"
conda activate env

linecount=0
while IFS=$'\t' read -r fields; do
    # echo $fields
    read -a field_arr <<< "$fields"
    # Check if the line starts with a #
    if [[ $linecount -eq 0 ]]; then
        # Header row
        read -a argnames <<< "$fields"
        nargs=${#argnames[@]}
        # echo $nargs
    elif [[ ! ${field_arr[0]} =~ ^# ]]; then
        # Process each field in the line
        cmd="python $rundir/tmp_script.py"
        for ((i=0; i<$nargs; i++)); do
            argname=${argnames[$i]}
            argval=${field_arr[$i]}
            if [[ "$argval" == "True" ]]; then
                argval=""
            fi
            cmd="$cmd --$argname $argval"
        done
        echo $cmd
        eval $cmd
    fi
    ((linecount++))
done < "$argfile"

rm $rundir/tmp_script.py
echo "Done!"
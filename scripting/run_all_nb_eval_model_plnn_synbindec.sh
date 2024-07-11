#!/bin/bash

rundir=notebooks
argfile=scripting/arglist_nb_eval_model_plnn_synbindec.tsv

logoutfile=scripting/logs/log_nb_eval_model_plnn_synbindec.o

echo STDOUT LOG: > $logoutfile

jupyter nbconvert --RegexRemovePreprocessor.patterns="^%" \
    --to script $rundir/nb_eval_model_plnn_synbindec.ipynb \
    --output tmp_script

eval "$(conda shell.bash hook)"
conda activate env

linecount=0
while IFS=$'\t' read -r fields; do
    read -a field_arr <<< "$fields"
    # Check if the line starts with a #
    if [[ $linecount -eq 0 ]]; then
        # Header row
        read -a argnames <<< "$fields"
        nargs=${#argnames[@]}
    elif [[ ! ${field_arr[0]} =~ ^# ]]; then
        # Process each field in the line
        cmd="python -u $rundir/tmp_script.py"
        for ((i=0; i<$nargs; i++)); do
            argname=${argnames[$i]}
            argval=${field_arr[$i]}
            if [[ "$argval" == "True" ]]; then
                argval=""
            fi
            cmd="$cmd --$argname $argval"
        done
        echo $cmd
        echo \\n$cmd\\n >> $logoutfile
        eval $cmd 2>&1 | tee -a $logoutfile
    fi
    ((linecount++))
done < "$argfile"

rm $rundir/tmp_script.py
echo "Done!"
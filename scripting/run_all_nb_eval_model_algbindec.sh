#!/bin/bash

sh scripting/_run_all_nb_eval_model_script.sh nb_eval_model_algbindec

# rundir=notebooks
# argfile=scripting/arglist_nb_eval_model_algbindec.tsv

# logoutfile=scripting/logs/log_nb_eval_model_algbindec.o
# logstatusfile=scripting/logs/log_nb_eval_model_algbindec.status

# echo STDOUT LOG: > $logoutfile
# echo "OUTPUT STATUS LOG (directory exitcode):" > $logstatusfile

# jupyter nbconvert --RegexRemovePreprocessor.patterns="^%" \
#     --to script $rundir/nb_eval_model_algbindec.ipynb \
#     --output tmp_script_nb_eval_model_algbindec

# eval "$(conda shell.bash hook)"
# conda activate env

# linecount=0
# while IFS=$'\t' read -r fields; do
#     read -a field_arr <<< "$fields"
#     # Check if the line starts with a #
#     if [[ $linecount -eq 0 ]]; then
#         # Header row
#         read -a argnames <<< "$fields"
#         nargs=${#argnames[@]}
#     elif [[ ! ${field_arr[0]} =~ ^# ]]; then
#         # Process each field in the line
#         cmd="python -u $rundir/tmp_script_nb_eval_model_algbindec.py"
#         for ((i=0; i<$nargs; i++)); do
#             argname=${argnames[$i]}
#             argval=${field_arr[$i]}
#             if [[ "$argname" == "modeldir" ]]; then
#                 modeldir=$argval
#             elif [[ "$argname" == "basedir" ]]; then
#                 basedir=$argval
#             fi
#             if [[ "$argval" == "True" ]]; then
#                 argval=""
#             fi
#             if [[ "$argval" != "False" ]]; then
#                 cmd="$cmd --$argname $argval"
#             fi
#         done
#         echo $cmd
#         echo \\n$cmd\\n >> $logoutfile
#         $cmd 2>&1 | tee -a $logoutfile
#         echo $basedir/$modeldir ${PIPESTATUS[0]} >> $logstatusfile
#     fi
#     ((linecount++))
# done < "$argfile"

# rm $rundir/tmp_script_nb_eval_model_algbindec.py
# echo "Done!"
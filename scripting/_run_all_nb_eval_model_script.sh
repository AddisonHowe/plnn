#!/usr/bin/env bash

# Function to check if a value is an integer
is_integer() {
    [[ "$1" =~ ^-?[0-9]+$ ]]
}

# Check the number of arguments
if [ $# -ne 1 ] && [ $# -ne 3 ]; then
    echo "Usage: $0 <notebook_basename> [<start> <stop>]"
    exit 1
fi

nb_basename=$1  # example: nb_eval_model_algbindec

# Check if second and third arguments are integers if provided
if [ $# -eq 3 ]; then
    if ! is_integer "$2" || ! is_integer "$3"; then
        echo "Error: The second and third arguments must be integers."
        exit 1
    fi
    start_idx=$2
    stop_idx=$3
    logoutfile=scripting/logs/log_${nb_basename}_"$2"-"$3".o
    logstatusfile=scripting/logs/log_${nb_basename}_"$2"-"$3".status
else
    start_idx=0
    stop_idx=1000000
    logoutfile=scripting/logs/log_${nb_basename}.o
    logstatusfile=scripting/logs/log_${nb_basename}.status
fi

rundir=notebooks
argfile=scripting/arglist_${nb_basename}.tsv

echo STDOUT LOG: > $logoutfile
echo "OUTPUT STATUS LOG (directory exitcode):" > $logstatusfile

tmp_script_dir=scripting/tmp
tmp_script_fname=tmp_script_${nb_basename}_"$start_idx"-"$stop_idx"
tmp_script_fpath=$tmp_script_dir/$tmp_script_fname.py

mkdir -p $tmp_script_dir
jupyter nbconvert --RegexRemovePreprocessor.patterns="^%" \
    --to script $rundir/${nb_basename}.ipynb \
    --output-dir $tmp_script_dir --output $tmp_script_fname

eval "$(conda shell.bash hook)"
conda activate env

mapfile -t lines < "$argfile"
linecount=0
nprocessed=0
for line in "${lines[@]}"; do
    # Split the line into fields using tab as the delimiter
    IFS=$'\t' read -r -a field_arr <<< "$line"
    if [[ $linecount -eq 0 ]]; then
        # Header row
        read -a argnames <<< "$line"
        nargs=${#argnames[@]}
    elif [[ ! ${field_arr[0]} =~ ^# ]]; then
        if [ $nprocessed -ge "$start_idx" ] && [ $nprocessed -lt "$stop_idx" ]; then
            # Process each field in the line
            cmd="python -u $tmp_script_fpath"
            for ((i=0; i<$nargs; i++)); do
                argname=${argnames[$i]}
                argval=${field_arr[$i]}
                if [[ "$argname" == "modeldir" ]]; then
                    modeldir=$argval
                elif [[ "$argname" == "basedir" ]]; then
                    basedir=$argval
                fi
                if [[ "$argval" == "True" ]]; then
                    argval=""
                fi
                if [[ "$argval" != "False" ]]; then
                    cmd="$cmd --$argname $argval"
                fi
            done
            echo $cmd
            printf "\n$cmd\n" >> $logoutfile
            $cmd 2>&1 | tee -a $logoutfile
            echo $basedir/$modeldir ${PIPESTATUS[0]} >> $logstatusfile
        fi
        ((nprocessed++));
    fi
    ((linecount++))
done

rm $tmp_script_fpath
echo "Done!"
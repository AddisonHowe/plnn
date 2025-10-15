#!/usr/bin/env bash

datdir_orig="tests/_data/test_main_data"
outdirbase="tests/_data/test_main_data3d"
pyscript=tests/_data/transform_data.py
transform=transform1

# mkdir -p ${outdirbase}
# cp -r ${datdir_orig}/* ${outdirbase}

for k in training_data validation_data; do
    subdirs=$(ls ${outdirbase}/${k})
    for sd in ${subdirs[@]}; do
        dpath=${outdirbase}/${k}/${sd}
        opath=${outdirbase}/${k}/${sd}
        if [[ -d $dpath ]]; then
            echo $sd
            python $pyscript -i ${dpath}/xs.npy -o ${opath}/xs.npy -t transform1
        fi
    done
done

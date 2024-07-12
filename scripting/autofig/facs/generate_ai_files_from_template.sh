#!/bin/bash

# Location of Adobe Illustrator
ILLUSTRATOR_PATH="/Applications/Adobe Illustrator 2023/Adobe Illustrator.app"

# Base project directory, using tilde without expansion as explained below.
PROJ_DIR_TILDE="~/Documents/Projects/plnn"
PROJ_DIR="${PROJ_DIR_TILDE/#\~/$HOME}"

# Output filename suffix, to append to the model name
IMAGE_SUFFIX=image1

# Template .ai file, and the folder containing images linked in the template.
# Note that the folder name will be replaced, and therefore needs to use the
# tilde explicitly, without substitution, for the filename.
template_fpath=$PROJ_DIR/scripting/autofig/facs/facs_template1.ai
template_linkdir=$PROJ_DIR_TILDE/scripting/autofig/facs/template1_images

# This is where all generated ai files will be stored, one for every run below.
aioutdir=${PROJ_DIR}/figures/out/autofig/facs

# Script to modify the links in an .ai file, with placeholder in/out files,
# and temporary generated script, with placeholder in/out files replaced
scriptfpath=$PROJ_DIR/scripting/autofig/modify_links.jsx
tmp_script_fpath=$PROJ_DIR/scripting/autofig/facs/_tmp_modify_links.jsx

# Directories containing images corresponding to trained models.
rundirs=(
    out/eval_models_facs/model_facs_dec1a_2dnmf_v1_20240624_135342
    out/eval_models_facs/model_facs_dec1a_2dpca_v1_20240624_133245
    out/eval_models_facs/model_facs_dec1a_2dpca_v1_20240627_133353
    out/eval_models_facs/model_facs_dec1a_2dpca_v1_20240627_163619
    out/eval_models_facs/model_facs_dec1a_2dpca_v1_20240627_165606
    out/eval_models_facs/model_facs_dec1a_2dpca_v1_20240627_193058
    out/eval_models_facs/model_facs_dec1a_2dpca_v2_20240627_193058
    out/eval_models_facs/model_facs_dec1b_2dnmf_v1_20240624_135122
    out/eval_models_facs/model_facs_dec1b_2dpca_v1_20240624_133245
    out/eval_models_facs/model_facs_dec1b_2dpca_v1_20240627_193058
    out/eval_models_facs/model_facs_dec1b_2dpca_v2_20240627_193058
    out/eval_models_facs/model_facs_dec1b_2dpca_v3_20240705_153205
    out/eval_models_facs/model_facs_dec1b_2dpca_v5_20240710_202109
    out/eval_models_facs/model_facs_dec1b_2dpca_v6_20240710_202109
    out/eval_models_facs/model_facs_dec1b_2dpca_v7_20240710_201912
    out/eval_models_facs/model_facs_dec1b_2dpca_v10_20240711_153302
    out/eval_models_facs/model_facs_dec1b_2dpca_v11_20240711_141418
    out/eval_models_facs/model_facs_dec2a_2dnmf_v1_20240624_135342
    out/eval_models_facs/model_facs_dec2a_2dpca_v1_20240624_133245
    out/eval_models_facs/model_facs_dec2b_2dnmf_v1_20240624_135140
    out/eval_models_facs/model_facs_dec2b_2dpca_v1_20240624_133245
    out/eval_models_facs/model_facs_dec2b_2dpca_v5_20240710_222319
    out/eval_models_facs/model_facs_dec2b_2dpca_v6_20240710_222319
    out/eval_models_facs/model_facs_dec2b_2dpca_v7_20240710_222319
    out/eval_models_facs/model_facs_dec2b_2dpca_v8_20240710_205933
    out/eval_models_facs/model_facs_dec2b_2dpca_v8_20240710_215556
    out/eval_models_facs/model_facs_v2_dec1b_2dpca_v1_20240711_143632
    out/eval_models_facs/model_facs_v2_dec1b_2dpca_v2_20240711_143632
    out/eval_models_facs/model_facs_v2_dec1b_2dpca_v3_20240711_144144
    out/eval_models_facs/model_facs_v2_dec1b_2dpca_v4_20240711_145314
    out/eval_models_facs/model_facs_v2_dec2b_2dpca_v5_20240712_121934
    out/eval_models_facs/model_facs_v2_dec2b_2dpca_v6_20240712_121955
    out/eval_models_facs/model_facs_v2_dec2b_2dpca_v7_20240712_122158
    out/eval_models_facs/model_facs_v2_dec2b_2dpca_v8_20240712_121611
)

# Main Loop
for rd in ${rundirs[@]}; do
    modelname=$(basename $rd)
    echo $modelname
    fname=${modelname}_${IMAGE_SUFFIX}
    cp $template_fpath $aioutdir/$fname.ai
    open -a "$ILLUSTRATOR_PATH" $aioutdir/$fname.ai
    sed -e "s|<OLD_FOLDER_PATH>|$template_linkdir|" \
        -e "s|<NEW_FOLDER_PATH>|$PROJ_DIR/$rd|" $scriptfpath > $tmp_script_fpath
    osascript -e 'tell application "Adobe Illustrator" to do javascript file "'"$tmp_script_fpath"'"';
    rm $tmp_script_fpath
    rm $aioutdir/$fname.ai  # remove .ai file, keeping only the pdf
done
echo Done!
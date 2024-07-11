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
template_fpath=$PROJ_DIR/scripting/autofig/synbindec/synbindec_template1.ai
template_linkdir=$PROJ_DIR_TILDE/scripting/autofig/synbindec/template1_images

# This is where all generated ai files will be stored, one for every run below.
aioutdir=${PROJ_DIR}/figures/out/autofig/synbindec

# Script to modify the links in an .ai file, with placeholder in/out files,
# and temporary generated script, with placeholder in/out files replaced
scriptfpath=$PROJ_DIR/scripting/autofig/modify_links.jsx
tmp_script_fpath=$PROJ_DIR/scripting/autofig/synbindec/_tmp_modify_links.jsx

# Directories containing images corresponding to trained models.
rundirs=(
    out/eval_models_plnn_synbindec/model_phi1_1a_v_kl1_20240705_024039
    out/eval_models_plnn_synbindec/model_phi1_1a_v_kl2_20240705_042608
    out/eval_models_plnn_synbindec/model_phi1_1a_v_kl3_20240705_050801
    out/eval_models_plnn_synbindec/model_phi1_1a_v_kl4_20240705_054948
    out/eval_models_plnn_synbindec/model_phi1_1a_v_kl5_20240705_085735
    # out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd1_20240627_193208
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd1_20240704_134102
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd1_fix_noise_large_20240704_134101
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd1_fix_noise_small_20240704_134102
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd1_largenet_20240704_134101
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd1_presample_20240704_134101
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd1_smallnet_20240704_134101
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd2_20240704_134547
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd3_20240704_134627
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd4_20240704_140023
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd5_20240704_142037
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd6_20240705_132548
    out/eval_models_plnn_synbindec/model_phi1_1a_v_mmd7_20240708_191953
    out/eval_models_plnn_synbindec/model_phi2_1a_v_mmd1_20240704_142345
    out/eval_models_plnn_synbindec/model_phiq_1a_v_mmd1_20240704_143517
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
done
echo Done!
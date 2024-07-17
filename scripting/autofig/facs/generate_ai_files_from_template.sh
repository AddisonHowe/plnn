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
rundirs=$(ls out/eval_models_facs/)

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
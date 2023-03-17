#!/bin/bash

src=$(pwd)
output=output
subject_list=$src/available_subjects_ASDPRSPCA.csv
Jacobians=Jacobians
template=$src/week40_T2w.nii.gz
all_Jacobians=all_Jacobians.nii.gz
mask=$src/mask.nii.gz
regrid_mask=regrid_mask.nii.gz
ID=()

. ../FBA/support_functions.sh

mkdir -p $src/$output

cd $src/$output

while read p; do
	sub_ses=$(echo $p | awk -F, '{print $1}' | awk -F'/' '{print $1"_"$2}')
	ID+=$(echo "$src/$Jacobians/${sub_ses}_logjacs_in_extdhcp40wks_smooth3sigmavox.nii.gz ")
done < $subject_list

#creating 40 weeks mask
#bet $template $mask -o -m -R -f 0.1
run 'Merge them together' \
	mrcat $(IN $ID) OUT:$all_Jacobians

run 'regrid mask' \
	mrgrid IN:$mask regrid -template IN:$all_Jacobians OUT:$regrid_mask

#generate design and contrast matrix using Glm GUI
run 'generate contrast design matrix' \
	python $script/generate_ID_list.py matrix \
	--file IN:$src/$subject_list --sep , \
	--continuous 2 3 5 6 7 8\
	--contnames GA PMA ASDPRS AncP1 AncP2 AncP3 \
	--categorical 1 \
	--catnames sex \
	--contrast ASDPRS \
	--out_ID OUT:id_files.txt \
	--out_design OUT:design_matrix.txt \
	--out_contrast OUT:contrast_matrix.txt

#
#run 'Perform randomise' \
#	randomise -i IN:test_mergeJacobians.nii.gz -o OUT-name:test_stats -m $regrid_mask -
#
#run 'Merge the Jacobians together' \
#	mrcat $(IN $ID) OUT:mergedJacobians.nii.gz
#

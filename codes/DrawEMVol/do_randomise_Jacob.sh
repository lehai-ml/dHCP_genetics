#!/bin/bash

src=$(pwd)
output=output_Jacobians
jacobians_clusterstat=jacobians_clusterstat
subjects_list=$src/Jacobians_volume_ASDPCA_after_ancestry_outliers.txt
template=$src/week40_T2w.nii.gz
mask=mask_gm_wm_only.mif
ID=()
output_folder=cortical_only_4sigma_no_TBV
. ../FBA/scripts/support_functions.sh

mkdir -p $src/$output

cd $output
#
#run 'Perform randomise' \
#	randomise -i IN:test_mergeJacobians.nii.gz -o OUT-name:test_stats -m $regrid_mask -
#
#run 'Merge the Jacobians together' \
#	mrcat $(IN $ID) OUT:mergedJacobians.nii.gz

##!/bin/bash
#doing mrclusterstats

ID_list=()
while read subj; do
if [ "x$subj" == "x" ]; then continue; fi
if [[ "$subj" =~ ^[[:space:]]*# ]]; then continue; fi
IFS=',' read -ra subj <<< $subj
ID_list+=(${subj[0]})
done < $subjects_list

run 'regrid mask'  \
	mrgrid IN:$src/$mask regrid -voxel 1 OUT:regrided_mask.mif

for ID in "${ID_list[@]}"; do
(

echo '###################################'
echo '    '$ID
echo '###################################'

id_ses=$(echo $ID | sed 's/\//_/')
file_to_check=$src/dafnis/Jacobians_in_extdhcp40wks/${id_ses}_logjacs_in_extdhcp40wks_smooth3sigmavox.nii.gz
run 'regrid log jacobians to parcellation atlas' \
	mrgrid $file_to_check regrid -template IN:regrided_mask.mif OUT:${id_ses}_regrided_log_jacobians.mif

run 're smooth jacobians' \
	mrfilter IN:${id_ses}_regrided_log_jacobians.mif smooth -stdev 4 OUT:${id_ses}_regrided_log_jacobians_smoothed.mif
) || continue
done

pt=( ASD_PC1 )
id_file=id_files_smoothed.txt

run 'getting contrast and design matrices' \
  python ../../FBA/generate_ID_list.py matrix \
  --file IN:$subjects_list --sep , \
  --categorical 1 \
  --catnames sex \
  --intercept \
  --standardize \
  --continuous 2 3 5 6 7 8 \
  --contnames GA PMA ${pt[@]} AncPC1 AncPC2 AncPC3 \
  --contrast ${pt[@]} \
  --id_suffix _regrided_log_jacobians_smoothed \
  --out_ID OUT:$id_file
#you can change if you want to work with jacobians or smoothed jacobians.
pt=( ASD_PC1 )
for pt in ${pt[@]}; do
    echo "###########################"
    echo "doing ${pt}"
    echo "###########################"
    if [ ! -d "$output_folder/$pt" ]; then
	mkdir -p $output_folder/$pt
        mrclusterstats $id_file $pt"_design.txt" $pt"_contrast.txt" regrided_mask.mif $output_folder/$pt/stat
    fi
done
cp $id_file $output_folder/$pt/stat
cp $pt"_design.txt" $output_folder/$pt/stat
cp $pt"_contrast.txt" $output_folder/$pt/stat


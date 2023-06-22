#!/bin/bash
#doing mrclusterstats

mkdir -p $src/$output_folder/$jacobians_clusterstat

for ID in "${ID_list[@]}"; do
(

echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID

if [[ -d $src/$individual_fods_output/$ID ]]; then
    check_here="$src/$individual_fods_output/$ID/"
fi


id_ses=$(echo $ID | sed 's/\//_/')

file_to_check=$log_jacobians/${id_ses}_logjacs_in_extdhcp40wks_smooth3sigmavox.nii.gz
run 'regrid log jacobians to wm parcellation atlas' \
	mrgrid $file_to_check regrid -template IN:$check_here$subject_wm_parc_in_40wks OUT:$check_here$regrided_log_jacobians

run 're smooth jacobians' \
	mrfilter IN:$check_here$regrided_log_jacobians smooth -stdev 5 OUT:$check_here$regrided_log_jacobians_smoothed

run 'regrid log jacobains to warped wm mask 1.3 voxel ' \
	mrgrid IN:$check_here$regrided_log_jacobians_smoothed regrid -template IN:$src/$output_folder/$warped_mask_average OUT:$src/$output_folder/$jacobians_clusterstat/$id_ses$"_"$regrided_log_jacobians_in_warped_wm_mask_template 
) || continue
done

cd $src/$output_folder/$jacobians_clusterstat

pt=( ASD_imputed_Pt_1em8 ASD_imputed_Pt_1em6 ASD_imputed_Pt_1em5 ASD_imputed_Pt_00001 ASD_imputed_Pt_0001 ASD_imputed_Pt_001 ASD_imputed_Pt_005 ASD_imputed_01 ASD_imputed_Pt_05 ASD_imputed_Pt_all ASD_imputed_PC1 ASD_imputed_CS )
id_file=id_file_imputed.txt

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix \
  --file IN:$src/$subjects_list --sep , \
  --categorical 1 \
  --catnames sex \
  --intercept \
  --standardize \
  --continuous 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --contnames GA PMA TBV ${pt[@]} AncPC1 AncPC2 AncPC3 \
  --contrast ${pt[@]} \
  --id_suffix "_"${regrided_log_jacobians_in_warped_wm_mask_template%".mif"} \
  --out_ID OUT:$id_file


#pt=( ASD_PRS_Pt_001 )
for pt in ${pt[@]}; do
    echo "###########################"
    echo "doing ${pt}"
    echo "###########################"
    if [ ! -d "whole_brain/$pt" ]; then
	mkdir -p whole_brain/$pt
        mrclusterstats $id_file $pt"_design.txt" $pt"_contrast.txt" $src/$output_folder/$warped_mask_average whole_brain/$pt/stat
    fi
done

#if [ ! -d "test" ];
#
#run 'running mrclusterstat' \
#    mrclusterstats IN:$id_file IN:$design_matrix IN:$contrast_matrix IN:$src/$output_folder/$warped_mask_average OUT:test  

#!/bin/bash
#this file perform simple VBA- utilises fsl randomise
# mrclusterstat took too long to run :( check with donald
# if it has been fixed

mkdir -p $src/$output_folder/$vba

for ID in ${ID_list[@]}; do
(

echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID
id_ses=$(echo $ID | sed 's/\//_/')
#measure=( FA )
#run 'transforming FA images to joint atlas space' \
#    mrtransform 
#
#run 'transforming FA images to joint atlas space' \
#    mrtransform IN:$dt_fa -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk -interp cubic - \| mrconvert - OUT:$src/$output_folder/$dti_stats/$DTI_in_template_space/${id_ses}_fa.nii.gz 
#

#run 'registering individual FOD to WM FOD parcellation space' \
#    mrregister IN:$wm_norm_fod IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$mask -nl_warp OUT:$native2wm_parc_warp OUT:$wm_parc2native_warp
#
#run 'transforming wm parcellation to subject space' \
#    mrtransform IN:$wm_parcellation_by_Alena -warp IN:$wm_parc2native_warp -interp nearest OUT:${subject_wm_parc}
#
#dti_to_measure=( fa )
#for dti in ${dti_to_measure[@]} ; do
#    mean_value=()
#    for tract in {94..147}; do
#        mean_value+=$(mrcalc $subject_wm_parc $tract -eq - | mrstats dt_$dti.mif -mask - -output mean)
#    done
#    echo $id_ses ${mean_value[@]} >> $src/$output_folder/$dti_stats/mean_$dti.txt
#done
#
)||continue
done

cd $src/$output_folder/$dti_stats

#run 'getting contrast and design matrices' \
#  python $src/generate_ID_list.py matrix \
#  --file IN:$src/$subjects_list --sep , \
#  --categorical 1 \
#  --catnames sex \
#  --continuous 2 3 6 7 8 9 \
#  --contnames PMA GA PRSPCA AncPC1 AncPC2 AncPC3 \
#  --contrast PRSPCA \
#  --no-standardize \
#  --no-intercept \
#  --sort_id \
#  --generate_vest \
#  --out_ID OUT:$stats_folder/$id_file \
#  --out_design OUT:$stats_folder/$design_matrix \
#  --out_contrast OUT:$stats_folder/$contrast_matrix

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix \
  --file IN:$src/$subjects_list --sep , \
  --continuous 2 \
  --contnames PMA \
  --contrast PMA \
  --no-standardize \
  --no-intercept \
  --sort_id \
  --generate_vest \
  --out_ID OUT:$id_file \
  --out_design OUT:$design_matrix \
  --out_contrast OUT:$contrast_matrix

#run 'concatenating into 4D nii.gz image (note to sort id)'\
#    mrcat $(IN $DTI_in_template_space/*) OUT:all_FA.nii.gz
#
#run 'generate WM mask ' \
#    mrgrid IN:$src/$output_folder/$output_5TT/$wm regrid -template IN:all_FA.nii.gz -interp nearest - \| mrthreshold - -abs 50 - \| mrconvert - OUT:regrided_wm_mask.nii.gz
#
#run 'perform FSL VBA' \
#    randomise -i IN:all_FA.nii.gz -o OUT-name:ICV -d IN:design.mat -t IN:design.con -m IN:regrided_wm_mask.nii.gz -T -n 1000

##run 'perform Clusterstats' \
##   mrgrid IN:$src/$output_folder/$output_5TT/$wm regrid -template IN:$src/$output_folder/$warped_wm_fod_average -interp nearest - \| mrthreshold - -abs 50 - \| mrclusterstats IN:$id_file IN:$design_matrix IN:$contrast_matrix - OUT-name:test -notest
#

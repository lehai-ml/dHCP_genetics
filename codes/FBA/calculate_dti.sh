#!/bin/bash
#this file perform simple VBA- utilises fsl randomise
# mrclusterstat took too long to run :( check with donald
# if it has been fixed

mkdir -p $src/$output_folder/$dti_stats/$DTI_in_template_space

for ID in ${ID_list[@]}; do
(

echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID
run 'extracting DTI' \
  dwiextract IN:$src/$dwi_data/$ID/$dwi -shells 0,1000 - \| dwi2tensor -mask IN:$mask - OUT:$diffusion_tensor

run 'calculating FA' \
  tensor2metric IN:$diffusion_tensor -fa OUT:$dt_fa

#run 'calculating ADC' \
#  tensor2metric IN:$diffusion_tensor -adc OUT:$dt_adc
#
#run 'calculating RD' \
#  tensor2metric IN:$diffusion_tensor -rd OUT:$dt_rd
#
#run 'calculating AD' \
#  tensor2metric IN:$diffusion_tensor -ad OUT:$dt_ad

#
id_ses=$(echo $ID | sed 's/\//_/')
#
#run 'transforming FA images to joint atlas space' \
#    mrtransform IN:$dt_fa -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk -interp cubic - \| mrconvert - OUT:$src/$output_folder/$dti_stats/$DTI_in_template_space/${id_ses}_fa.nii.gz 
#

run 'registering individual FOD to WM FOD parcellation space' \
    mrregister IN:$wm_norm_fod IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$mask -nl_warp OUT:$native2wm_parc_warp OUT:$wm_parc2native_warp

run 'transforming wm parcellation to subject space' \
    mrtransform IN:$wm_parcellation_by_Alena -warp IN:$wm_parc2native_warp -interp nearest OUT:${subject_wm_parc}

dti_to_measure=( fa )
for dti in ${dti_to_measure[@]} ; do
    mean_value=()
    for tract in {94..147}; do
        mean_value+=$(mrcalc $subject_wm_parc $tract -eq - | mrstats dt_$dti.mif -mask - -output mean)
    done
    echo $id_ses ${mean_value[@]} >> $src/$output_folder/$dti_stats/mean_$dti.txt
done

)||continue
done

#cd $src/$output_folder/$dti_stats

#run 'getting contrast and design matrices' \
#  python $src/generate_ID_list.py matrix \
#  --file IN:$src/$subjects_list --sep , \
#  --continuous 4 \
#  --contnames ICV  \
#  --contrast ICV \
#  --standardize \
#  --id_prefix $DTI_in_template_space/ \
#  --id_suffix _fa \
#  --sort_id \
#  --out_ID OUT:$id_file \
#  --out_design OUT:$design_matrix \
#  --out_contrast OUT:$contrast_matrix
#
#run 'generating FSL design matrix' \
#    Text2Vest IN:$design_matrix OUT:design.mat
#
#run 'generating FSL contrast matrix' \
#    Text2Vest IN:$contrast_matrix OUT:design.con
#
#run 'concatenating into 4D nii.gz image (note to sort id)'\
#    mrcat $(IN $DTI_in_template_space/*) OUT:all_FA.nii.gz
#
#run 'generate WM mask ' \
#    mrgrid IN:$src/$output_folder/$output_5TT/$wm regrid -template IN:all_FA.nii.gz -interp nearest - \| mrthreshold - -abs 50 - \| mrconvert - OUT:regrided_wm_mask.nii.gz
#
#run 'perform FSL VBA' \
#    randomise -i IN:all_FA.nii.gz -o OUT-name:ICV -d IN:design.mat -t IN:design.con -m IN:regrided_wm_mask.nii.gz -T -n 1000
#
##run 'perform Clusterstats' \
##   mrgrid IN:$src/$output_folder/$output_5TT/$wm regrid -template IN:$src/$output_folder/$warped_wm_fod_average -interp nearest - \| mrthreshold - -abs 50 - \| mrclusterstats IN:$id_file IN:$design_matrix IN:$contrast_matrix - OUT-name:test -notest
#

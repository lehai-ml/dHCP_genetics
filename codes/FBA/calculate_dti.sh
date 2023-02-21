#!/bin/bash

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

run 'transforming FA images to joint atlas space' \
    mrtransform IN:$dt_fa -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk -interp cubic - \| mrconvert - OUT:$src/$output_folder/$tbss/$DTI_in_template_space/${id_ses}_fa.nii.gz 
#
#run 'transforming FA images to average template' \
#    mrtransform IN:$dt_fa -warp IN:$native2average_warp OUT:$src/$output_folder/$dti_stats/$DTI_in_template_space/${id_ses}_fa.mif
#
)||continue
done


#mkdir -p $src/$output_folder/$dti_stats
#
#cd $src/$output_folder/$dti_stats
#
#run 'getting contrast and design matrices' \
#  python $src/generate_ID_list.py matrix \
#  --file IN:$src/$subjects_list --sep , \
#  --categorical 1 \
#  --catnames sex \
#  --continuous 2 3 6 7 8 9\
#  --contnames PMA GA SCZPCA euro_Anc_PC1 euro_Anc_PC2 euro_Anc_PC3\
#  --standardize \
#  --contrast SCZPCA \
#  --id_prefix $DTI_in_template_space/ \
#  --id_suffix _fa \
#  --out_ID OUT:$id_file \
#  --out_design OUT:$design_matrix \
#  --out_contrast OUT:$contrast_matrix
#
#run 'perform Clusterstats' \
#   mrgrid IN:$src/$output_folder/$output_5TT/$wm regrid -template IN:$src/$output_folder/$warped_wm_fod_average -interp nearest - \| mrclusterstats IN:$id_file IN:$design_matrix IN:$contrast_matrix - OUT:FA_stats_

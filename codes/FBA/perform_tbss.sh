#!/bin/bash

dwi=postmc_dstriped-dwi.nii.gz
bval=postmc_dstriped-dwi.bval
bvec=postmc_dstriped-dwi.bvec
bet_mask=mask_T2w_brainmask_processed.nii.gz

mkdir -p $output_folder/$tbss
cd $output_folder/$tbss
for ID in ${ID_list[@]};do
(
echo '##########################'
echo '    '$ID
echo '##########################'
mkdir -p $ID
cd $ID

echo $pwd
echo $src/$dwi_data/$ID/$dwi

dtifit -k $src/$dwi_data/$ID/$dwi -m $src/$dwi_data/$ID/$bet_mask -r $src/$dwi_data/$ID/$bvec -b $src/$dwi_data/$ID/$bval -o dti
)
done
#for measure in ${measures[@]}; do
#    mkdir -p $tbss/$measure
#    for ID in ${ID_list[@]}; do
#	id_ses=$(echo $ID | sed 's/\//_/')
#        run 'converting to fsl nii.gz' \
#		mrconvert IN:$ID/dt_$measure.mif OUT:$tbss/$measure/${id_ses}_${measure}.nii.gz
#    done
#done
#
#cd $tbss/fa
#mrconvert $src/$output_folder/$warped_wm_fod_average warped_wm_fod_average.nii.gz
#tbss_1_preproc *.nii.gz

#tbss_2_reg -t warped_wm_fod_average.nii.gz
tbss_3_postreg -S

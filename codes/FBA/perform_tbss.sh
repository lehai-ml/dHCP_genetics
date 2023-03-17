#!/bin/bash

dwi=postmc_dstriped-dwi.nii.gz
bval=postmc_dstriped-dwi.bval
bvec=postmc_dstriped-dwi.bvec
bet_mask=mask_T2w_brainmask_processed.nii.gz
ID_template=ID_template.txt
all_subjects_dti=all_subjects_dti.txt
DTI_TK_processed=DTI_TK_processed
stats_folder=stats

#######Template variables#######
mean_initial_template=mean_initial_template.nii.gz

mkdir -p $output_folder/$tbss/$DTI_TK_processed
cd $output_folder/$tbss
for ID in "${ID_list[@]}";do
(
echo '##########################'
echo '    '$ID
echo '##########################'
mkdir -p $ID
cd $ID
id_ses=$(echo $ID | sed 's/\//_/')

run 'calculate DTI using FSL' \
	dtifit -k IN:$src/$dwi_data/$ID/$dwi -m IN:$src/$dwi_data/$ID/$bet_mask -r IN:$src/$dwi_data/$ID/$bvec -b IN:$src/$dwi_data/$ID/$bval -o OUT-name:dti

if [ ! -f dti_dtitk.nii.gz ]; then
   echo 'convert to DTI-TK compatible format'
   fsl_to_dtitk dti
fi
if [ ! -f dti.nii.gz ]; then
   echo 'turn all DWI files into a single file in DTI space'
   TVFromEigenSystem -basename dti -type FSL
   TVAdjustVoxelspace -in dti.nii.gz -origin 0 0 0
fi

run 'Perform SPD - make tensor symmetric positive and definite' \
   TVtool -in IN:dti.nii.gz -spd -out OUT:dti_spd.nii.gz

run 'Changing units to match dtitk units' \
   TVtool -in IN:dti_spd.nii.gz -scale 1000 -out OUT:dti_unit_matched.nii.gz

run 'Changing voxel to be powers of two' \
    TVResample -in IN:dti_unit_matched.nii.gz -out OUT:$src/$output_folder/$tbss/$DTI_TK_processed/${id_ses}_dti_res.nii.gz -align center -size 128 128 64

)
done

cd $DTI_TK_processed

run 'Create Template' \
    TVMean -in IN:$ID_template -out OUT:$mean_initial_template

echo 'generate list of all subjects'
ls *_dti_res.nii.gz > $all_subjects_dti

if [ ! -f ${all_subjects_dti%.txt}_aff.txt ]; then
    echo "perform rigid-body alignment"
    dti_rigid_population $mean_initial_template $all_subjects_dti EDS 3
    echo "perform affine alignment"
    dti_affine_population mean_rigid3.nii.gz $all_subjects_dti EDS 3
fi

run 'create mask dtitky style' \
    TVtool -in IN:mean_affine3.nii.gz -tr -out OUT:mean_affine3_mask.nii.gz

run 'binarise the mask' \
    mrcalc IN:mean_affine3_mask.nii.gz 0.01 -gt mean_affine3_mask.nii.gz 100 -lt -mult OUT:mask.nii.gz

if [ ! -f mean_diffeomorphic_initial6.nii.gz ]; then
    echo 'performing last registration'
    dti_diffeomorphic_population mean_affine3_mask.nii.gz ${all_subjects_dti%.txt}_aff.txt mask.nii.gz 0.002
    echo 'warping individual images to template'
    dti_warp_to_template_group $all_subjects_dti mean_diffeomorphic_initial6.nii.gz 1.5 1.5 1.5
    #this command will create a file called *_dti_res_diffeo.nii.gz
    #this is weird because the result output here is the same (?) as *_dti_res_aff_diffeo.nii.gz and the dti_res_diffeo.nii.gz is not used subsequently?
fi

echo 'the subjects are now registered and need to re-integrate into TBSS'

run 'generating template FA map step 1' \
    TVMean -in IN:${all_subjects_dti%.txt}_aff_diffeo.txt -out OUT:mean_final_high_res.nii.gz

run 'generating template FA map step 2' \
    TVtool -in IN:mean_final_high_res.nii.gz -fa -out OUT:mean_FA.nii.gz

echo "calculating individual FA, AD, RD and TR"
measures=( FA ) #AD RD or TR
for measure in ${measures[@]}; do
    mkdir -p ${measure}
    while read p; do
	echo "====================="
	echo "${p%_dti*}" 
	echo "====================="
	run "calculating ${measure}" \
		TVtool -in IN:$p -${measure,,} -out OUT:$measure/${p%_dti*}_$measure.nii.gz
    done < ${all_subjects_dti%.txt}_aff_diffeo.txt
done
echo ""
echo ""
echo ""
echo "========================================="
echo "           TBSS                          "
echo "========================================="
echo ""
echo ""
echo ""
run 'generating FA skeleton from DTI template' \
    tbss_skeleton -i IN:mean_FA.nii.gz -o OUT:mean_FA_skeleton.nii.gz


for measure in ${measures[@]}; do
    run "Combine each ${measure} map into a 4D volume" \
	mrcat $(IN $measure/sub-*) OUT:$measure/all_${measure}.nii.gz
    #NOTE: doing concat like this will sort the ids in the increasing order.
    run "Generating mask for mean images " \
	fslmaths IN:$measure/all_${measure}.nii.gz -max 0 -Tmin -bin OUT:$measure/mean_${measure}_mask.nii.gz -odt char
done    

cd $src/$output_folder/$tbss
mkdir -p $stats_folder/


cp -u $src/$output_folder/$tbss/$DTI_TK_processed/mean_FA_skeleton.nii.gz $src/$output_folder/$tbss/$stats_folder/

cp -u $src/$output_folder/$tbss/$DTI_TK_processed/mean_FA.nii.gz $src/$output_folder/$tbss/$stats_folder/

cp -u $src/$output_folder/$tbss/$DTI_TK_processed/FA/all_FA.nii.gz $src/$output_folder/$tbss/$stats_folder/

cp -u $src/$output_folder/$tbss/$DTI_TK_processed/FA/mean_FA_mask.nii.gz $src/$output_folder/$tbss/$stats_folder/


if [ ! -f $stats_folder/mean_FA_skeleton_mask.nii.gz ]; then
    tbss_4_prestats 0.1
fi






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
#tbss_3_postreg -S

#!/bin/bash


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
	dtifit -k IN:$src/$dwi_data/$ID/$dwi_nii -m IN:$src/$dwi_data/$ID/$bet_mask -r IN:$src/$dwi_data/$ID/$bvec -b IN:$src/$dwi_data/$ID/$bval -o OUT-name:dti

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

echo "calculating individual FA"

measures=( AD TR RD )
for measure in "${measures[@]}"; do
mkdir -p $src/$output_folder/$tbss/$measure
    while read p; do
	echo "====================="
	echo "${p%_dti*}" 
	echo "====================="
	run "calculating ${measure} " \
	    TVtool -in IN:$p -${measure,,} -out OUT:$src/$output_folder/$tbss/$measure/${p%_dti*}.nii.gz
    done < ${all_subjects_dti%.txt}_aff_diffeo.txt
done
echo ""
echo ""
echo ""
echo "========================================="
echo "           TBSS-FA                       "
echo "========================================="
echo ""
echo ""
echo ""

cd $src/$output_folder/$tbss
mkdir -p $stats_folder

cp -u $src/$output_folder/$tbss/$DTI_TK_processed/mean_FA.nii.gz $src/$output_folder/$tbss/$stats_folder/

run 'generating FA skeleton from DTI template' \
    tbss_skeleton -i IN:$stats_folder/mean_FA.nii.gz -o OUT:$stats_folder/mean_FA_skeleton.nii.gz

run "Combine each FA map into a 4D volume" \
    mrcat $(IN FA/sub-*) OUT:$stats_folder/all_FA.nii.gz
#NOTE: doing concat like this will sort the ids in the increasing order.
run "Generating mask for mean images " \
    fslmaths IN:$stats_folder/all_FA.nii.gz -max 0 -Tmin -bin OUT:$stats_folder/mean_FA_mask.nii.gz -odt char

if [ ! -f $stats_folder/mean_FA_skeleton_mask.nii.gz ]; then
    tbss_4_prestats 0.1
fi


echo ""
echo ""
echo ""
echo "========================================="
echo "           TBSS-non-FA                   "
echo "========================================="
echo ""
echo ""
echo ""

cd $src/$output_folder/$tbss/

measures=( AD TR RD MD )
for measure in "${measures[@]}";do
if [ $measure == "MD" ]; then
    run 'generating all MD' \
	fslmaths IN:$stats_folder/all_TR.nii.gz -div 3 OUT:$stats_folder/all_MD.nii.gz
else
run "Combine each $measure into a 4D volume" \
	mrcat $(IN $measure/sub-*) OUT:$stats_folder/all_$measure.nii.gz
fi
run "Generating $measure skeleton "\
	tbss_skeleton -i IN:$stats_folder/mean_FA.nii.gz -p 0.15 IN:$stats_folder/mean_FA_skeleton_mask_dst.nii.gz ~/fsl/data/standard/LowerCingulum_1mm IN:$stats_folder/all_$measure.nii.gz OUT:$stats_folder/all_${measure}_skeletonised.nii.gz

done

# generate design matrices

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix \
  --file IN:$src/$subjects_list --sep , \
  --categorical 1 \
  --catnames sex \
  --no-standardize \
  --continuous 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
  --contnames GA PMA TBV ASD_PRS_Pt_1em08 ASD_PRS_Pt_1em06 ASD_PRS_Pt_1em5 ASD_PRS_Pt_00001 ASD_PRS_Pt_0001 ASD_PRS_Pt_001 ASD_PRS_Pt_005 ASD_PRS_Pt_01 ASD_PRS_Pt_05 ASD_PRS_Pt_all ASD_PRS_PC1 AncPC1 AncPC2 AncPC3 \
  --contrast ASD_PRS_Pt_1em08 ASD_PRS_Pt_1em06 ASD_PRS_Pt_1em5 ASD_PRS_Pt_00001 ASD_PRS_Pt_0001 ASD_PRS_Pt_001 ASD_PRS_Pt_005 ASD_PRS_Pt_01 ASD_PRS_Pt_05 ASD_PRS_Pt_all ASD_PRS_PC1 \
  --sort_id \
  --no-intercept \
  --generate_vest \
  --out_ID OUT:$stats_folder/$id_file \

#  
#run 'getting contrast and design matrices' \
#  python $src/generate_ID_list.py matrix \
#  --file IN:$src/$subjects_list --sep , \
#  --categorical 1 \
#  --catnames sex \
#  --continuous 2 3 5 6 7 8 \
#  --contnames GA PMA ASD_PRS_Pt_001 AncPC1 AncPC2 AncPC3 \
#  --contrast ASD_PRS_Pt_001 \
#  --no-standardize \
#  --no-intercept \
#  --sort_id \
#  --generate_vest \
#  --out_ID OUT:$stats_folder/$id_file \
#  --out_design OUT:$stats_folder/$design_matrix \
#  --out_contrast OUT:$stats_folder/$contrast_matrix
#

cd $stats_folder

threshold=( ASD_PRS_Pt_1em08 ASD_PRS_Pt_1em06 ASD_PRS_Pt_1em5 ASD_PRS_Pt_00001 ASD_PRS_Pt_0001 ASD_PRS_Pt_001 ASD_PRS_Pt_005 ASD_PRS_Pt_01 ASD_PRS_Pt_05 ASD_PRS_Pt_all ASD_PRS_PC1 )
threshold=( ASD_PRS_Pt_001 )
for pt in ${threshold[@]}; do
    echo "###########################"
    echo "doing ${pt}"
    echo "###########################"
    if [ ! -d "$pt" ]; then
	mkdir -p $pt
	randomise -i all_FA -o $pt -d $pt"_"design.txt -t $pt"_"contrast.txt -m mean_FA_skeleton_mask -n 1000 --T2 -D
    fi
done

#randomise -i all_FA -o ASD_PRS_Pt_001 -d designmatrix.txt -t contrast_matrix -m mean_FA_skeleton_mask -n 1000 --T2 -D


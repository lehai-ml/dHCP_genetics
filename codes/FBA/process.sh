#!/bin/bash


# Relevant files to access
src="$(pwd)"
dwi_data=data
dwi=postmc_dstriped-dwi300.mif
bet_mask=mask_T2w_brainmask_processed.nii.gz
warps=dhcp_neo_dMRI_derived
warps_in_40wk=from-dmrishard_to-extdhcp40wk_mode-image.mif.gz
wm_response=dHCP_atlas_v2.1_rf_wm.dhsfa015_44 # this response function is generated from 21 WM response functions from subjects aged 44.1 weeks
csf_response=dHCP_atlas_v2.1_rf_csf.dhsfa015 # this response function is generated by averaging all CSF response functions.
subjects_list=subjects_list.txt
output_folder=output

mask=mask.mif
wm_fod=wm_fod.mif
csf_fod=csf_fod.mif

wm_norm_fod=wm_norm_fod.mif
csf_norm_fod=csf_norm_fod.mif

warped_mask_in_dHCP_40wk=warped_mask_in_dHCP_40wk.mif
warped_wm_fod_in_dHCP_40wk=warped_wm_fod_in_dHCP_40wk.mif

warped_mask_average=warped_mask_average.mif
warped_wm_fod_average=warped_wm_fod_average.mif

native2average_warp=native2average_warp.mif
average2native_warp=average2native_warp.mif

fixel_mask=fixel_mask

fod_in_template_space_NOT_REORIENTED=fod_in_template_space_NOT_REORIENTED.mif
fixel_in_template_space_NOT_REORIENTED=fixel_in_template_space_NOT_REORIENTED
fd=fd.mif
fixel_in_template_space=fixel_in_template_space

all_subj_fd=all_subj_fd 
all_subj_fc=all_subj_fc
all_subj_log_fc=all_subj_log_fc
all_subj_fdc=all_subj_fdc
set -e
. support_functions.sh

. generate_ID_list.sh


# Start of main script:

if [ ! -d data ]; then
    echo "dhcp-pipeline-data not yet mounted"
    sudo mount -t cifs //isi01/dhcp-pipeline-data /home/lh20/dhcp-pipeline-data/ -o username=lh20,domain=isd,iocharset=utf8,vers=2.1
    echo "dhcp-pipeline-data mounted"
fi


# build list of subjects:
ID_list=()
while read subj; do
  # skip if empty:
  if [ "x$subj" == "x" ]; then continue; fi

  # split line into components to get folder name:
  IFS=',' read -ra subj <<< $subj
  ID_list+=(${subj[0]})
done < $subjects_list


for ID in ${ID_list[@]}; do 
(

echo '###################################'
echo '    '$ID
echo '###################################'

mkdir -p $output_folder/$ID
cd $output_folder/$ID
run 'testing estimating mask' \
  dwi2mask IN:$src/$dwi_data/$ID/$dwi - \| mrcalc IN:$src/$dwi_data/$ID/$bet_mask - -multiply OUT:$mask

run 'estimating fiber orientations distributions (FODs)' \
  dwi2fod msmt_csd IN:$src/$dwi_data/$ID/$dwi -mask IN:$mask IN:$src/$warps/$wm_response OUT:$wm_fod IN:$src/$warps/$csf_response OUT:$csf_fod

run 'normalising FODs' \
  mtnormalise IN:$wm_fod OUT:$wm_norm_fod IN:$csf_fod OUT:$csf_norm_fod -mask IN:$mask

id_ses=$(echo $ID | sed 's/\//_/')
run 'warping mask to joint atlas space' \
  mrtransform IN:$mask -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk  OUT:$warped_mask_in_dHCP_40wk -interp linear

run 'warping normalised FODs to joint atlas space' \
  mrtransform IN:$wm_norm_fod -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk OUT:$warped_wm_fod_in_dHCP_40wk -reorient_fod yes -interp cubic


) || continue
done



#creating average image of 40 week masks and FODs
cd $output_folder
echo '###################################'
echo ''
echo ''
subject_masks=$(find ${ID_list[@]} -name $warped_mask_in_dHCP_40wk)
subject_fods=$(find ${ID_list[@]} -name $warped_wm_fod_in_dHCP_40wk)

run 'generating an average image of 40 weeks masks' \
	echo IN:$subjects_list \| mrmath $(IN $subject_masks) min -datatype bit - \| mrgrid - regrid -voxel 1.3 -interp nearest OUT:$warped_mask_average

run 'generating an average image of 40 weeks FODs' \
	echo IN:$subjects_list \| mrmath $(IN $subject_fods) mean - \| mrgrid - regrid -voxel 1.3 -interp sinc OUT:$warped_wm_fod_average

update_folder_if_needed run "'calculating fixel from average mask'" \
	fod2fixel -mask IN:$warped_mask_average -fmls_peak_value 0.06 IN:$warped_wm_fod_average OUT:$fixel_mask



echo ''
echo ''
echo '###################################'

#Registering FODs to average templates
cd $src

for ID in ${ID_list[@]}; do 
(
echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID

run 'registering normalised FODs to average images' \
  mrregister IN:$wm_norm_fod -mask1 IN:$mask IN:$src/$output_folder/$warped_wm_fod_average -nl_warp OUT:$native2average_warp OUT:$average2native_warp

run 'transforming FOD images to average template' \
  mrtransform IN:$wm_norm_fod -warp IN:$native2average_warp -reorient_fod no OUT:$fod_in_template_space_NOT_REORIENTED

update_folder_if_needed run "'segmenting FOD images to estimate fixels and their fiber density'" \
 fod2fixel -mask IN:$src/$output_folder/$warped_mask_average IN:$fod_in_template_space_NOT_REORIENTED OUT:$fixel_in_template_space_NOT_REORIENTED -afd OUT:$fd

update_folder_if_needed run "'reorienting direction of fixels'" \
 fixelreorient IN:$fixel_in_template_space_NOT_REORIENTED IN:$native2average_warp OUT:$fixel_in_template_space

cd $src/$output_folder
id_ses=$(echo $ID | sed 's/\//_/')
update_folder_if_needed run "'running fixel correspondence and computing fiber density'"\
 fixelcorrespondence IN:$ID/$fixel_in_template_space/$fd IN:$fixel_mask OUT:$all_subj_fd OUT:${id_ses}.mif

update_folder_if_needed run "'computing fibre cross-section'" \
 warp2metric IN:$ID/$native2average_warp -fc IN:$fixel_mask OUT:$all_subj_fc OUT:${id_ses}.mif

mkdir -p $all_subj_log_fc
run 'computing log of fibre cross-section' \
 mrcalc IN:$all_subj_fc/${id_ses}.mif -log OUT:$all_subj_log_fc/${id_ses}.mif

mkdir -p $all_subj_fdc
run 'computing combined measure FDC' \
 mrcalc IN:$all_subj_fd/${id_ses}.mif IN:$all_subj_fc/${id_ses}.mif -mult OUT:$all_subj_fdc/${id_ses}.mif

)||continue
done

#copying direction and index.mif to log_fc and fdc files 
cp $output_folder/$all_subj_fd/{directions.mif,index.mif} $output_folder/$all_subj_log_fc
cp $output_folder/$all_subj_fd/{directions.mif,index.mif} $output_folder/$all_subj_fdc

# perform whole brain tracktography on the FOD template
cd $src
. 5tt.sh
cd $src
. tractography.sh
# reduce biases in tractogram densities
# smooth the data

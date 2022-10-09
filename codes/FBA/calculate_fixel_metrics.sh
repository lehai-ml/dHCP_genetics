#!/bin/bash


sanity_check sub $output_folder/$all_subj_fd $output_folder/$all_subj_fc $output_folder/$all_subj_fdc $output_folder/$all_subj_log_fc

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

#!/bin/bash

cd $output_folder
echo '###################################'
echo ''
echo ''
subject_masks=$(find ${ID_list[@]} -name $warped_mask_in_dHCP_40wk)
subject_fods=$(find ${ID_list[@]} -name $warped_wm_fod_in_dHCP_40wk)

run 'generating an average image of 40 weeks masks' \
	echo IN:$src/$subjects_list \| mrmath $(IN $subject_masks) min -datatype bit - \| mrgrid - regrid -voxel 1.3 -interp nearest OUT:$warped_mask_average

run 'generating an average image of 40 weeks FODs' \
	echo IN:$src/$subjects_list \| mrmath $(IN $subject_fods) mean - \| mrgrid - regrid -voxel 1.3 -interp sinc OUT:$warped_wm_fod_average

update_folder_if_needed run "'calculating fixel from average mask'" \
	fod2fixel -mask IN:$warped_mask_average -fmls_peak_value 0.06 IN:$warped_wm_fod_average OUT:$fixel_mask

echo ''
echo ''
echo '###################################'


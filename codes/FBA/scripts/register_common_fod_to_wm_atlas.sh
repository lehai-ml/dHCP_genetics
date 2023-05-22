#!bin/bash
#the code is lifted from the perform_aba.sh

cd $output_folder

run 'registering common average wm_fod to WM FOD parcellation space' \
	mrregister IN:$warped_wm_fod_average IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$warped_mask_average -nl_warp OUT:$native2wm_parc_warp OUT:$wm_parc2native_warp

run 'transforming wm parcellation to average space' \
	mrtransform IN:$wm_parcellation_by_Alena -warp IN:$wm_parc2native_warp -interp nearest OUT:$wm_parcellation_warped_wm_fod

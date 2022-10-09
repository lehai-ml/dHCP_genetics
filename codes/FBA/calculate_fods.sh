#!/bin/bash

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


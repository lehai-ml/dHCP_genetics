#!/bin/bash
#this file perform Atlas-based statistics using Alena's neonatal white matter atlas.

for ID in ${ID_list[@]}; do
(
echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID

run 'registering individual FOD to WM FOD parcellation space' \
    mrregister IN:$wm_norm_fod IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$mask -nl_warp OUT:$native2wm_parc_warp OUT:$wm_parc2native_warp

run 'transforming wm parcellation to subject space' \
    mrtransform IN:$wm_parcellation_by_Alena -warp IN:$wm_parc2native_warp -interp nearest OUT:${subject_wm_parc}

)||continue
done

#
#dti_to_measure=( fa )
#for dti in ${dti_to_measure[@]} ; do
#    mean_value=()
#    for tract in {94..147}; do
#        mean_value+=$(mrcalc $subject_wm_parc $tract -eq - | mrstats dt_$dti.mif -mask - -output mean)
#    done
#    echo $id_ses ${mean_value[@]} >> $src/$output_folder/$dti_stats/mean_$dti.txt
#done

#!/bin/bash
#this file perform Atlas-based statistics using Alena's neonatal white matter atlas.

mkdir -p $output_folder/$aba

for ID in "${ID_list[@]}"; do
(
echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID

run 'registering individual FOD to WM FOD parcellation space' \
    mrregister IN:$wm_norm_fod IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$mask -nl_warp OUT:$native2wm_parc_warp OUT:$wm_parc2native_warp

run 'transforming wm parcellation to subject space' \
    mrtransform IN:$wm_parcellation_by_Alena -warp IN:$wm_parc2native_warp -interp nearest OUT:${subject_wm_parc}

id_ses=$(echo $ID | sed 's/\//_/')

to_measure=( fd )
for measure in ${to_measure[@]} ; do
    mean_output_file=$src/$output_folder/$aba/mean_${measure}.txt 
    if [ -f $mean_output_file ]; then
	if grep -Fq $id_ses $mean_output_file; then
	    continue
	fi
    fi
    mean_value=()
    if [ $measure == "fd" ];then
        file_to_check=wm_norm_fod.mif
	mrconvert $file_to_check -coord 3 0 -axes 0,1,2 tmp-$file_to_check
	for tract in {94..147}; do
	    mean_value+=$(mrcalc $subject_wm_parc $tract -eq - | mrstats tmp-$file_to_check -mask - -output mean)
	done
	echo $id_ses ${mean_value[@]} >> $mean_output_file
	rm tmp-$file_to_check
    elif [ $measure == "md" ] || [ $measure == "fa" ] || [ $measure == "ad" ]; then
	if [ $measure == "ad" ]; then measure=L1;fi
	file_to_check=$src/$output_folder/$tbss/$ID/dti_${measure^^}.nii.gz
	for tract in {94..147}; do
	    mean_value+=$(mrcalc $subject_wm_parc $tract -eq - | mrstats $file_to_check -mask - -output mean)
	done
	echo $id_ses ${mean_value[@]} >> $mean_output_file
    fi
done
)||continue
done

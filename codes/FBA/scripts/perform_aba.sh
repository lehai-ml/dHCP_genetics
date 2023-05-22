#!/bin/bash
#this file perform Atlas-based statistics using Alena's neonatal white matter atlas.

mkdir -p $output_folder/$aba

for ID in "${ID_list[@]}"; do
(
echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID

if [[ -d $src/$individual_fods_output/$ID ]]; then
    check_here="$src/$individual_fods_output/$ID/"
fi

id_ses=$(echo $ID | sed 's/\//_/')

run 'registering individual FOD to WM FOD parcellation space' \
    mrregister IN:$check_here$wm_norm_fod IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$check_here$mask -nl_warp OUT:$check_here$native2wm_parc_warp OUT:$check_here$wm_parc2native_warp

run 'transforming wm parcellation to subject space' \
    mrtransform IN:$wm_parcellation_by_Alena -warp IN:$check_here$wm_parc2native_warp -interp nearest OUT:$check_here${subject_wm_parc}

run 'transforming wm parcellation in subject space to 40 weeks commont emplate' \
    mrtransform IN:$check_here$subject_wm_parc -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk OUT:$check_here$subject_wm_parc_in_40wks -interp nearest

to_measure=( log_jacob )
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
	    mean_value+=$(mrcalc $check_heresubject_wm_parc $tract -eq - | mrstats tmp-$file_to_check -mask - -output mean)
	done
	echo $id_ses ${mean_value[@]} >> $mean_output_file
	rm tmp-$file_to_check
    elif [ $measure == "md" ] || [ $measure == "fa" ] || [ $measure == "ad" ]; then
	if [ $measure == "ad" ]; then measure=L1;fi
	file_to_check=$src/$output_folder/$tbss/$ID/dti_${measure^^}.nii.gz
	for tract in {94..147}; do
	    mean_value+=$(mrcalc $check_heresubject_wm_parc $tract -eq - | mrstats $file_to_check -mask - -output mean)
	done
	echo $id_ses ${mean_value[@]} >> $mean_output_file
    elif [ $measure == "log_jacob" ]; then
        file_to_check=$log_jacobians/${id_ses}_logjacs_in_extdhcp40wks_smooth3sigmavox.nii.gz
	run 'regrid log jacobians' \
	    mrgrid $file_to_check regrid -template $check_here$subject_wm_parc_in_40wks OUT:$check_here$regrided_log_jacobians
	run 're smooth jacobians' \
	    mrfilter IN:$check_here$regrided_log_jacobians smooth -stdev 5 OUT:$check_here$regrided_log_jacobians_smoothed
	for tract in {94..147};do
	    mean_value+=$(mrcalc $check_here$subject_wm_parc_in_40wks $tract -eq - | mrstats $check_here$regrided_log_jacobians_smoothed -mask - -output mean)
	done
	echo $id_ses ${mean_value[@]} >> $mean_output_file
    fi
done
)||continue
done

#!/bin/bash
#this file perform Atlas-based statistics using Alena's neonatal white matter atlas.

mkdir -p $output_folder/$aba

#how to do linear interpolation
#first generate binary mask for each of the mask
#then perform linear interpolation on each of the binary tract
#then look at voxel in each volume
# take argmax for each voxel.


# THIS PIECE IS TO DO TRANSFORM WM PARCELLATION TO INDIVIDUAL SPACE #
# however, what i am doing instead is transform WM parcellation to the group-study space #
###### Transform WM parcellation to group-study space ###
#for ID in "${ID_list[@]}"; do
#(
#echo '###################################'
#echo '    '$ID
#echo '###################################'
#
#cd $output_folder/$ID
#
#if [[ -d $src/$individual_fods_output/$ID ]]; then
#    check_here="$src/$individual_fods_output/$ID/"
#fi
#
#id_ses=$(echo $ID | sed 's/\//_/')
#
#run 'registering individual FOD to WM FOD parcellation space' \
#    mrregister IN:$check_here$wm_norm_fod IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$check_here$mask -nl_warp OUT:$check_here$native2wm_parc_warp OUT:$check_here$wm_parc2native_warp
#
#run 'transforming wm parcellation to subject space' \
#    mrtransform IN:$wm_parcellation_by_Alena -warp IN:$check_here$wm_parc2native_warp -interp linear OUT:$check_here${subject_wm_parc}
#
#run 'transforming wm parcellation in subject space to 40 weeks common template' \
#    mrtransform IN:$check_here$subject_wm_parc -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk OUT:$check_here$subject_wm_parc_in_40wks -interp linear
#
#)||continue
#done
######################

run 'Registering group average FOD to the common template ODF' \
    mrregister IN:$output_folder/$warped_wm_fod_average IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$output_folder/$warped_mask_average -nl_warp OUT:$output_folder/$aba/$native2wm_parc_warp OUT:$output_folder/$aba/$wm_parc2native_warp

### Previously have been doing with nearest neighbour but MARIA recommend to use linear interpolation instead ####
run 'Transforming the wm parcellation to the group average space using the nearest neighbour' \
    mrtransform IN:$wm_parcellation_by_Alena -warp IN:$output_folder/$aba/$wm_parc2native_warp -interp nearest OUT:$output_folder/$aba/$wm_parcellation_in_average_fod_space_nearest

### DOING it with linear interpolation

run 'Transforming the wm parcellation to the group average space using the linear interpolation' \
    python perform_linear_interpolation.py --WM-parcellation IN:$wm_parcellation_by_Alena --group-FOD IN:$output_folder/$warped_wm_fod_average --nl-warp IN:$output_folder/$aba/$wm_parc2native_warp --output-file OUT:$output_folder/$aba/wm_parcellation_in_average_fod_space_linear.nii

run 'transform the nii file to mif file' mrconvert IN:$output_folder/$aba/wm_parcellation_in_average_fod_space_linear.nii OUT:$output_folder/$aba/$wm_parcellation_in_average_fod_space_linear

###GENERATING FIXEL MASK for each tract  ##################
###It shouldn't matter what measures you put here, because it will only look at the index file and direction file.
measures=( fd )
wm_parcellation_in_average_fod_space=$src/$output_folder/$aba/$wm_parcellation_in_average_fod_space_linear
for measure in ${measures[@]};do
for tract in {94..147};do
    if [ ! -f $src/$output_folder/$aba/$tract_fixel_mask_aba/wm_${tract}_fixel.mif ]; then
    echo "generating fixel mask for tract ${tract}"
       mrcalc $wm_parcellation_in_average_fod_space $tract -eq - | voxel2fixel - $src/$output_folder/$output_tractography/$fba_output/all_subj_${measure}_smooth/ $src/$output_folder/$aba/$tract_fixel_mask_aba/ wm_${tract}_fixel.mif
    fi
done
done

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

to_measure=( log_fc fd fdc )
for measure in ${to_measure[@]} ; do
    mean_output_file=$src/$output_folder/$aba/mean_${measure}_not_smoothed_linear.txt 
    if [ -f $mean_output_file ]; then
	if grep -Fq $id_ses $mean_output_file; then
	    continue
	fi
    fi
    mean_value=()
    if [ $measure == "afd" ];then
        file_to_check=${check_here}wm_norm_fod.mif
	mrconvert $file_to_check -coord 3 0 -axes 0,1,2 tmp-${file_to_check#$check_here}
	for tract in {94..147}; do
	    mean_value+=$(mrcalc $check_here${subject_wm_parc} $tract -eq - | mrstats tmp-${file_to_check#$check_here} -mask - -output mean)
	done
	echo $id_ses ${mean_value[@]} >> $mean_output_file
	rm tmp-${file_to_check#$check_here}
    elif [ $measure == "md" ] || [ $measure == "fa" ] || [ $measure == "ad" ]; then
	if [ $measure == "ad" ]; then measure=L1;fi
	file_to_check=$src/$output_folder/$tbss/$ID/dti_${measure^^}.nii.gz
	for tract in {94..147}; do
	    mean_value+=$(mrcalc ${check_here}${subject_wm_parc} $tract -eq - | mrstats $file_to_check -mask - -output mean)
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
#    elif [ $measure == "log_fc" ] || [ $measure == "fdc" ] || [ $measure == "fd" ]; then
#	for tract in {94..147};do
#	    mean_value+=$(mrstats $src/$output_folder/$output_tractography/$fba_output/all_subj_${measure}_smooth/${id_ses}.mif -mask $src/$output_folder/aba/all_subj_tract_${measure}_smooth/wm_${tract}_fixel.mif -output mean) 
#        done
#	echo $id_ses ${mean_value[@]} >> $mean_output_file

    elif [ $measure == "log_fc" ] || [ $measure == "fdc" ] || [ $measure == "fd" ]; then
	echo "calculating the mean tract ${measure}"
	for tract in {94..147};do
	    mean_value+=$(mrstats $src/$output_folder/all_subj_${measure}/${id_ses}.mif -mask $src/$output_folder/$aba/$tract_fixel_mask_aba/wm_${tract}_fixel.mif -output mean) 
        done
	echo $id_ses ${mean_value[@]} >> $mean_output_file
    fi
done
)||continue
done

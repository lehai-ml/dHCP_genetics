#!/bin/bash
for ID in ${ID_list[@]}; do 
(

echo '###################################'
echo '    '$ID
echo '###################################'

mkdir -p $output_folder/$ID
cd $output_folder/$ID

if [[ -d $src/$individual_fods_output/$ID ]]; then
    check_here="$src/$individual_fods_output/$ID/"
fi

run 'combine estimating mask' \
  dwi2mask IN:$src/$dwi_data/$ID/$dwi - \| mrcalc IN:$src/$dwi_data/$ID/$bet_mask - -multiply OUT:$check_here$mask

run 'estimating fiber orientations distributions (FODs)' \
  dwi2fod msmt_csd IN:$src/$dwi_data/$ID/$dwi -mask IN:$check_here$mask IN:$src/$warps/$wm_response OUT:$check_here$wm_fod IN:$src/$warps/$csf_response OUT:$check_here$csf_fod

run 'normalising FODs' \
  mtnormalise IN:$check_here$wm_fod OUT:$check_here$wm_norm_fod IN:$csf_fod OUT:$check_here$csf_norm_fod -mask IN:$check_here$mask

id_ses=$(echo $ID | sed 's/\//_/')
run 'warping mask to joint atlas space' \
  mrtransform IN:$check_here$mask -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk  OUT:$check_here$warped_mask_in_dHCP_40wk -interp linear

run 'warping normalised FODs to joint atlas space' \
  mrtransform IN:$check_here$wm_norm_fod -warp IN:$src/$warps/$ID/$id_ses"_"$warps_in_40wk OUT:$check_here$warped_wm_fod_in_dHCP_40wk -reorient_fod yes -interp cubic


) || continue
done

cd $src
missing_files=($(diff -d $individual_fods_output $output_folder | grep "Only in ${output_folder}" | awk '{print $4}'))

if [ ${#missing_files[@]} -gt 0 ]; then

    echo -e "${RED}${#missing_files[@]} of the outputs in: "${output_folder}" are newly generated and are not found in the original output folder: "${individual_fods_output}"${NC}"
    echo -e "${RED}I will copy those files into the original output folder. So next time if you change the ID list, it won't rerun the same commands.${NC}"
    for folder in ${missing_files[@]}; do
        echo -e "copying ${RED}$output_folder/$folder to $individual_fods_output/ ${NC}"
	cp -r -u $output_folder/$folder $individual_fods_output/    
    done
    echo -e "${GREEN}Done.${NC}"

fi



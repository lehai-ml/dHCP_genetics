#!/bin/bash

echo "###################################"
echo "     "Performing Tractography
echo "###################################"

cd $src
mkdir -p $output_folder/$output_tractography/$fba_output
cd $output_folder/$output_tractography


#run 'creating streamlines' \
#  tckgen IN:$src/$output_folder/$warped_wm_fod_average -seed_image IN:$src/$output_folder/$warped_mask_average -mask IN:$src/$output_folder/$warped_mask_average -act IN:$src/$output_folder/$output_5TT/$image_5TT -select $number_of_streamlines -cutoff 0.06 -angle $angle -maxlen $maxlen -power $power OUT:$tracts
#
#run 'creating streamlines' \
#  tckgen IN:$src/$output_folder/$warped_wm_fod_average -seed_image IN:$src/$output_folder/$warped_mask_average -mask IN:$src/$output_folder/$warped_mask_average -act IN:$src/$output_folder/$output_5TT/$image_5TT -select 0 -seeds $number_of_streamlines OUT:$tracts
#length selection? tckedit?
run 'reducing number of streamlines' \
  tcksift IN:$tracts IN:$src/$output_folder/$warped_wm_fod_average OUT:$reduced_tracts -term_number $reduced_number_of_streamlines

update_folder_if_needed run "'generating fixel-fixel connectivity'" \
  fixelconnectivity IN:$src/$output_folder/$fixel_mask IN:$reduced_tracts OUT:$fba_output/$ffixel_matrix




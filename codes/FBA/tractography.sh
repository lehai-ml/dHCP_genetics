#!/bin/bash

echo "###################################"
echo "     "Performing Tractography
echo "###################################"
output_tractography=tractography
number_of_streamlines=100000
tracts="tracts_${number_of_streamlines}.tck"
reduced_number_of_streamlines=10000
reduced_tracts=reduced_tracts_${reduced_number_of_streamlines}.tck
fba_output=fba
ffixel_matrix=ffixel_matrix

all_subj_fd_smooth=all_subj_fd_smooth
all_subj_log_fc_smooth=all_subj_log_fc_smooth
all_subj_fdc_smooth=all_subj_fdc_smooth
cd $src
mkdir -p $output_folder/$output_tractography/$fba_output
cd $output_folder/$output_tractography
run 'creating streamlines' \
  tckgen IN:$src/$output_folder/$warped_wm_fod_average -seed_image IN:$src/$output_folder/$warped_mask_average -mask IN:$src/$output_folder/$warped_mask_average -act IN:$src/$output_folder/$output_5TT/$image_5TT -select 0 -seeds $number_of_streamlines OUT:$tracts


#length selection? tckedit?
run 'reducing number of streamlines' \
  tcksift IN:$tracts IN:$src/$output_folder/$warped_wm_fod_average OUT:$reduced_tracts -term_number $reduced_number_of_streamlines

update_folder_if_needed run "'generating fixel-fixel connectivity'" \
  fixelconnectivity IN:$src/$output_folder/$fixel_mask IN:$reduced_tracts OUT:$fba_output/$ffixel_matrix

update_folder_if_needed run "'smoothing FD data'" \
  fixelfilter IN:$src/$output_folder/$all_subj_fd smooth OUT:$fba_output/$all_subj_fd_smooth -matrix IN:$fba_output/$ffixel_matrix 
update_folder_if_needed run "'smoothing log FC data'" \
  fixelfilter IN:$src/$output_folder/$all_subj_log_fc smooth OUT:$fba_output/$all_subj_log_fc_smooth -matrix IN:$fba_output/$ffixel_matrix
update_folder_if_needed run "'smoothing FDC data'" \
  fixelfilter IN:$src/$output_folder/$all_subj_fdc smooth OUT:$fba_output/$all_subj_fdc_smooth -matrix IN:$fba_output/$ffixel_matrix

#run 'generating FBA IDs list'
  

#statistical analysis FD, log(FC) and FDC fixelfestats
#
#run 'calculating fixelcfestats FD' \
#  fixelcfestats $all_subj_fd_smooth files.txt design_matrix.txt contrast_matrix.txt $ffixel_matrix stats_fd/
#
#

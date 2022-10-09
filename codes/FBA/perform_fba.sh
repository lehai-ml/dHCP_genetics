#!bin/bash

cd $output_folder/$output_tractography

#run 'generating FBA IDs list'

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix --file IN:$src/$subjects_list --categorical 1 4 --continuous 2 3 --standardize --contrast 4 --catnames gender termness --contnames GA PMA --out_ID OUT:$fba_output/$id_file --out_design OUT:$fba_output/$design_matrix --out_contrast OUT:$fba_output/$contrast_matrix

sanity_check sub $fba_output/$id_file $src/$output_folder/$all_subj_fd $src/$output_folder/$all_subj_fc $src/$output_folder/$all_subj_log_fc $src/$output_folder/$all_subj_fdc

#update_folder_if_needed run "'smoothing FD data'" \
#  fixelfilter IN:$src/$output_folder/$all_subj_fd smooth OUT:$fba_output/$all_subj_fd_smooth -matrix IN:$fba_output/$ffixel_matrix 
#update_folder_if_needed run "'smoothing log FC data'" \
#  fixelfilter IN:$src/$output_folder/$all_subj_log_fc smooth OUT:$fba_output/$all_subj_log_fc_smooth -matrix IN:$fba_output/$ffixel_matrix
#update_folder_if_needed run "'smoothing FDC data'" \
#  fixelfilter IN:$src/$output_folder/$all_subj_fdc smooth OUT:$fba_output/$all_subj_fdc_smooth -matrix IN:$fba_output/$ffixel_matrix
#
##statistical analysis FD, log(FC) and FDC fixelfestats
##
#update_folder_if_needed run "'calculating fixelcfestats FD'" \
#  fixelcfestats IN:$fba_output/$all_subj_fd_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/$ffixel_matrix OUT:$fba_output/$stats_fd
#

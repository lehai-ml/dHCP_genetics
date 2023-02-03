#!bin/bash

cd $output_folder/$output_tractography

#run 'getting contrast and design matrices' \
#  python $src/generate_ID_list.py matrix \
#  --file IN:$src/$subjects_list --sep , \
#  --categorical 3 \
#  --catnames Gender \
#  --continuous 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
#  --contnames GA PMA \
#  PRS_1e-08 PRS_1e-07 PRS_1e-06 PRS_1e-05 PRS_0.0001 PRS_0.001 PRS_0.01 PRS_0.05 PRS_0.1 PRS_0.5 PRS_1 \
#  euro_Anc_PC1 euro_Anc_PC2 euro_Anc_PC3 \
#  --standardize \
#  --contrast PRS_0.1 PRS_0.5 PRS_1 \
#  --out_ID OUT:$fba_output/$id_file \
#  --out_design OUT:$fba_output/$design_matrix \
#  --out_contrast OUT:$fba_output/$contrast_matrix

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix \
  --file IN:$src/$subjects_list --sep , \
  --categorical 3 \
  --catnames sex \
  --continuous 1 2 \
  --contnames PMA GA \
  --standardize \
  --contrast GA \
  --out_ID OUT:$fba_output/$id_file \
  --out_design OUT:$fba_output/$design_matrix \
  --out_contrast OUT:$fba_output/$contrast_matrix

sanity_check sub $fba_output/$id_file $src/$output_folder/$all_subj_fd $src/$output_folder/$all_subj_fc $src/$output_folder/$all_subj_log_fc $src/$output_folder/$all_subj_fdc

update_folder_if_needed run "'smoothing FD data'" \
  fixelfilter IN:$src/$output_folder/$all_subj_fd smooth OUT:$fba_output/$all_subj_fd_smooth -matrix IN:$fba_output/$ffixel_matrix 
update_folder_if_needed run "'smoothing log FC data'" \
  fixelfilter IN:$src/$output_folder/$all_subj_log_fc smooth OUT:$fba_output/$all_subj_log_fc_smooth -matrix IN:$fba_output/$ffixel_matrix
update_folder_if_needed run "'smoothing FDC data'" \
  fixelfilter IN:$src/$output_folder/$all_subj_fdc smooth OUT:$fba_output/$all_subj_fdc_smooth -matrix IN:$fba_output/$ffixel_matrix

#statistical analysis FD, log(FC) and FDC fixelfestats
##
update_folder_if_needed run "'calculating fixelcfestats FD'" \
  fixelcfestats IN:$fba_output/$all_subj_fd_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/$ffixel_matrix OUT:$fba_output/$stats_fd

update_folder_if_needed run "'calculating fixelcfestats log FC'" \
  fixelcfestats IN:$fba_output/$all_subj_log_fc_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/$ffixel_matrix OUT:$fba_output/$stats_log_fc

update_folder_if_needed run "'calculating fixelcfestats FDC'" \
  fixelcfestats IN:$fba_output/$all_subj_fdc_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/$ffixel_matrix OUT:$fba_output/$stats_fdc



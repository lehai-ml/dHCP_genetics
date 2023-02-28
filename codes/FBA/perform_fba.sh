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

#run 'getting contrast and design matrices' \
#  python $src/generate_ID_list.py matrix \
#  --file IN:$src/$subjects_list --sep , \
#  --categorical 1 \
#  --catnames sex \
#  --continuous 2 3 6\
#  --contnames PMA GA C_allele\
#  --standardize \
#  --contrast C_allele \
#  --neg \
#  --out_ID OUT:$fba_output/$id_file \
#  --out_design OUT:$fba_output/$design_matrix \
#  --out_contrast OUT:$fba_output/$contrast_matrix
#
#sanity_check sub $fba_output/$id_file $src/$output_folder/$all_subj_fd $src/$output_folder/$all_subj_fc $src/$output_folder/$all_subj_log_fc $src/$output_folder/$all_subj_fdc
#
fba_measures_to_examine=( fd )

for fba_measure in ${fba_measures_to_examine[@]}; do
    update_folder_if_needed run "'smoothing $fba_measure data'" \
        fixelfilter IN:$src/$output_folder/all_subj_${fba_measure} smooth OUT:$fba_output/all_subj_${fba_measure}_smooth -matrix IN:$fba_output/$ffixel_matrix 
done

#statistical analysis FD, log(FC) and FDC fixelfestats
#
#for fba_measure in ${fba_measures_to_examine[@]}; do
#    update_folder_if_needed run "'calculating fixelcfestats $fba_measure'" \
#        fixelcfestats IN:$fba_output/all_subj_${fba_measure}_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/$ffixel_matrix OUT:$fba_output/stats_${fba_measure}
#    run 'copy design summary' copy_header IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix OUT:$fba_output/stats_${fba_measure}/$summary_contrast
#done
#

for fba_measure in ${fba_measures_to_examine[@]}; do
    update_folder_if_needed run "'calculating fixelcfestats $fba_measure -ftest only'" \
        fixelcfestats -ftests IN:$fba_output/ftest.txt -fonly IN:$fba_output/all_subj_${fba_measure}_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/${ffixel_matrix} OUT:$fba_output/stats_${fba_measure}
    run 'copy design summary' copy_header IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix OUT:$fba_output/stats_${fba_measure}/$summary_contrast
done


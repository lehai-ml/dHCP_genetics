#!bin/bash

cd $output_folder/$output_tractography

#run 'getting contrast and design matrices' \
#  python $src/generate_ID_list.py matrix \
#  --file IN:$src/$subjects_list --sep , \
#  --categorical 1 \
#  --catnames sex \
#  --standardize \
#  --continuous 2 3 4 5 6 7 8 9 10 11 12 \
#  --contnames PMA GA TBV PSM PSD DS AX HIS AncPC1 AncPC2 AncPC3 \
#  --contrast PSM PSD DS AX HIS \
#  --neg \
#  --intercept \
#  --out_ID OUT:$fba_output/$id_file \
#

#pt=( ASD_imputed_Pt_1em8 ASD_imputed_Pt_1em6 ASD_imputed_Pt_1em5 ASD_imputed_Pt_00001 ASD_imputed_Pt_0001 ASD_imputed_Pt_001 ASD_imputed_Pt_005 ASD_imputed_Pt_01 ASD_imputed_Pt_05 ASD_imputed_Pt_all ASD_imputed_PC1 ASD_imputed_CS )

pt=( ASD_Spark_Pt_1em8 ASD_Spark_Pt_1em6 ASD_Spark_Pt_1em5 ASD_Spark_Pt_00001 ASD_Spark_Pt_0001 ASD_Spark_Pt_001 ASD_Spark_Pt_005 ASD_Spark_Pt_01 ASD_Spark_Pt_05 ASD_Spark_Pt_all ASD_Spark_PC1 ASD_Spark_CS )

id_file=id_file_Spark.txt

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix \
  --file IN:$src/$subjects_list --sep , \
  --categorical 1 \
  --catnames sex \
  --standardize \
  --continuous 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --contnames GA PMA TBV ${pt[@]} AncPC1 AncPC2 AncPC3 \
  --contrast ${pt[@]} \
  --intercept \
  --out_ID OUT:$fba_output/$id_file \


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
fba_measures_to_examine=( log_fc )

for fba_measure in ${fba_measures_to_examine[@]}; do
    update_folder_if_needed run "'smoothing $fba_measure data'" \
        fixelfilter IN:$src/$output_folder/all_subj_${fba_measure} smooth OUT:$fba_output/all_subj_${fba_measure}_smooth -matrix IN:$fba_output/$ffixel_matrix 
done

#statistical analysis FD, log(FC) and FDC fixelfestats
#pt=( ASD_PRS_Pt_001 ASD_PRS_Pt_1em8 ASD_PRS_Pt_1em6 ASD_PRS_Pt_1em5 ASD_PRS_Pt_00001 ASD_PRS_Pt_0001 ASD_PRS_Pt_005 ASD_PRS_Pt_01 ASD_PRS_Pt_05 ASD_PRS_Pt_all ASD_PRS_PC1 ASD_PRS_CS )
#pt=( PSD PSM DS AX HIS )
#pt=( ASD_Spark_Pt_001 )

pt=( ASD_Spark_Pt_001 ASD_Spark_Pt_005 ASD_Spark_Pt_01 ASD_Spark_Pt_05 ASD_Spark_Pt_all ASD_Spark_PC1 ASD_Spark_CS )

for fba_measure in ${fba_measures_to_examine[@]}; do
    for threshold in ${pt[@]}; do
	if [ ! -d "$fba_output/$threshold/stats_${fba_measure}" ]; then
            mkdir -p $fba_output/$threshold
	    design_matrix=$threshold"_design.txt"
	    contrast_matrix=$threshold"_contrast.txt"

    update_folder_if_needed run "'calculating fixelcfestats $fba_measure'" \
        fixelcfestats IN:$fba_output/all_subj_${fba_measure}_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/$ffixel_matrix OUT:$fba_output/$threshold/stats_${fba_measure}
        cp -u $fba_output/$design_matrix $fba_output/$threshold/stats_${fba_measure}/
        cp -u $fba_output/$contrast_matrix $fba_output/$threshold/stats_${fba_measure}/
	cp -u $fba_output/$id_file $fba_output/$threshold/stats_${fba_measure}/
	fi
    done
done

#for fba_measure in ${fba_measures_to_examine[@]}; do
#    update_folder_if_needed run "'calculating fixelcfestats $fba_measure -ftest only'" \
#        fixelcfestats -ftests IN:$fba_output/ftest.txt -fonly IN:$fba_output/all_subj_${fba_measure}_smooth IN:$fba_output/$id_file IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix IN:$fba_output/${ffixel_matrix} OUT:$fba_output/stats_${fba_measure}
#    run 'copy design summary' copy_header IN:$fba_output/$design_matrix IN:$fba_output/$contrast_matrix OUT:$fba_output/stats_${fba_measure}/$summary_contrast
#done
#

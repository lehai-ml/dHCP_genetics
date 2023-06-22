#!/bin/bash

#This file is used to perform either mean stats per tract per subject 
#or fixelcfestats per tract
mkdir -p $src/$output_folder/$output_tractography/$fba_output/
cd $src/$output_folder/$output_tractography

pt=( ASD_Spark_Pt_1em8 ASD_Spark_Pt_1em7 ASD_Spark_Pt_1em6 ASD_Spark_Pt1em5 ASD_Spark_Pt_00001 ASD_Spark_Pt_0001 ASD_Spark_Pt_001 ASD_Spark_Pt_005 ASD_Spark_Pt_01 ASD_Spark_Pt_05 ASD_Spark_Pt_all ASD_Spark_CS )

id_file=id_file_spark.txt

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix \
  --file IN:$src/$subjects_list --sep , \
  --categorical 1 \
  --catnames sex \
  --standardize \
  --continuous 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --contnames PMA GA TBV ${pt[@]} AncPC1 AncPC2 AncPC3 \
  --contrast ${pt[@]} \
  --neg \
  --intercept \
  --out_ID OUT:$fba_output/$id_file \

fba_measures_to_examine=( fdc log_fc fd )

#SMOOTHING ALREADY APPLIED TO THE WHOLE BRAIN
#for tract in ${tract_to_examine[@]};do
#    for fba_measure in ${fba_measures_to_examine[@]}; do
#	update_folder_if_needed run "'smoothing $fba_measure data'" \
#	    fixelfilter IN:$src/$output_folder/all_subj_${fba_measure} smooth OUT:$fba_output/all_subj_${fba_measure}_smooth -matrix IN:$fba_output/${tract}_${ffixel_matrix} 
#    done
#done
#
cd $src/$output_folder/$output_tractography/$individual_tracts
####BE CAREFUL TO CHECK IF YOU HAVE SORTED THE IDS#######
for tract in ${tract_to_examine[@]};do
    run "generating fixel mask for $tract" \
        tck2fixel IN:$tract/$tract.tck \
	IN:$src/$output_folder/$fixel_mask \
	OUT:$tract/$tract_fixel_mask \
	OUT:${tract}_fixel.mif
    done

for tract in ${tract_to_examine[@]};do
    for fba_measure in ${fba_measures_to_examine[@]}; do
	 run "calculate individual mean $fba_measure for $tract" \
	    while read subj\; do mrstats IN:$src/$output_folder/$output_tractography/$fba_output/all_subj_${fba_measure}_smooth/\$subj -mask IN: $tract/$tract_fixel_mask/${tract}_fixel.mif -output mean\; done \< IN:$src/$output_folder/$output_tractography/$fba_output/$id_file \> OUT:$tract/mean-${fba_measure}-${tract}.txt
    done
done
####BE CAREFUL TO CHECK IF YOU HAVE SORTED THE IDS#######

#pt=( ASD_PRS_Pt_001 ASD_PRS_CS ASD_PRS_PC1 )
tract_to_examine=( $cst_L_R )
for tract in ${tract_to_examine[@]};do
    for fba_measure in ${fba_measures_to_examine[@]}; do 
	for threshold in ${pt[@]};do
	    echo $threshold
	    if [ ! -d "$tract/$threshold/stats_${fba_measure}" ]; then
		mkdir -p $tract/$threshold
	    fi
	    design_matrix=$src/$output_folder/$output_tractography/$fba_output/$threshold"_design.txt"
	    contrast_matrix=$src/$output_folder/$output_tractography/$fba_output/$threshold"_contrast.txt"
	    tract_output_fba_folder=$tract/$threshold/${tract}_stats${fba_measure}


	    update_folder_if_needed run "'performing $fba_measure analysis CFEstat for tract $tract'" \
		fixelcfestats IN:$src/$output_folder/$output_tractography/$fba_output/all_subj_${fba_measure}_smooth \
		IN:$src/$output_folder/$output_tractography/$fba_output/$id_file \
		IN:$design_matrix \
		IN:$contrast_matrix \
		IN:$src/$output_folder/$output_tractography/$fba_output/${tract}_${ffixel_matrix} \
		-mask IN:$tract/$tract_fixel_mask/${tract}_fixel.mif \
		OUT:$tract_output_fba_folder
	    cp -u $design_matrix $tract_output_fba_folder
	    cp -u $contrast_matrix $tract_output_fba_folder
	    cp -u $src/$output_folder/$output_tractography/$fba_output/$id_file $tract_output_fba_folder
	done
    done
done

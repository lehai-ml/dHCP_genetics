#!/bin/bash

#This file is used to perform either mean stats per tract per subject 
#or fixelcfestats per tract
mkdir -p $src/$output_folder/$output_tractography/$fba_output/
cd $src/$output_folder/$output_tractography

run 'getting contrast and design matrices' \
  python $src/generate_ID_list.py matrix \
  --file IN:$src/$subjects_list --sep , \
  --categorical 1 \
  --catnames sex \
  --continuous 2 3 4 5 6 7 8\
  --contnames GA PMA ASD_PRS_Pt_001 TBV AncPC1 AncPC2 AncPC3\
  --standardize \
  --contrast ASD_PRS_Pt_001 \
  --intercept \
  --out_ID OUT:$fba_output/$id_file \
  --out_design OUT:$fba_output/$design_matrix \
  --out_contrast OUT:$fba_output/$contrast_matrix

fba_measures_to_examine=( log_fc fdc fd )

#for tract in ${tract_to_examine[@]};do
#    for fba_measure in ${fba_measures_to_examine[@]}; do
#	update_folder_if_needed run "'smoothing $fba_measure data'" \
#	    fixelfilter IN:$src/$output_folder/all_subj_${fba_measure} smooth OUT:$fba_output/all_subj_${fba_measure}_smooth -matrix IN:$fba_output/${tract}_${ffixel_matrix} 
#    done
#done
#
cd $src/$output_folder/$output_tractography/$individual_tracts

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

for tract in ${tract_to_examine[@]};do
    for fba_measure in ${fba_measures_to_examine[@]}; do 
        update_folder_if_needed run "'performing $fba_measure analysis CFEstat for tract $tract'" \
            fixelcfestats IN:$src/$output_folder/$output_tractography/$fba_output/all_subj_${fba_measure}_smooth \
	    IN:$src/$output_folder/$output_tractography/$fba_output/$id_file \
	    IN:$src/$output_folder/$output_tractography/$fba_output/$design_matrix \
	    IN:$src/$output_folder/$output_tractography/$fba_output/$contrast_matrix \
	    IN:$src/$output_folder/$output_tractography/$fba_output/${tract}_${ffixel_matrix} \
	    -mask IN:$tract/$tract_fixel_mask/${tract}_fixel.mif \
	    OUT:$tract/${tract}_stats${fba_measure}
	run 'copy design summary' copy_header IN:$src/$output_folder/$output_tractography/$fba_output/$design_matrix \
	    IN:$src/$output_folder/$output_tractography/$fba_output/$contrast_matrix \
	    OUT:$tract/${tract}_stats${fba_measure}/$summary_contrast
    done
done

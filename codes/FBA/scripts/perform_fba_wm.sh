#!/bin/bash

#This file is used to perform either mean stats per tract per subject 
#or fixelcfestats per tract
cd $src/$output_folder/$output_tractography/$individual_tracts

for tract in ${tract_to_examine[@]};do
    run "generating fixel mask for $tract" \
        tck2fixel IN:$tract/$tract.tck \
	IN:$src/$output_folder/$fixel_mask \
	OUT:$tract/$tract_fixel_mask \
	OUT:${tract}_fixel.mif
    done

fba_measures_to_examine=( fd fdc log_fc )

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
	    IN:$src/$output_folder/$output_tractography/$fba_output/$ffixel_matrix \
	    -mask IN:$tract/$tract_fixel_mask/${tract}_fixel.mif \
	    OUT:$tract/${tract}_stats${fba_measure}
	run 'copy design summary' copy_header IN:$src/$output_folder/$output_tractography/$fba_output/$design_matrix \
	    IN:$src/$output_folder/$output_tractography/$fba_output/$contrast_matrix \
	    OUT:$tract/${tract}_stats${fba_measure}/$summary_contrast
    done
done

#!/bin/bash

#This file is used to generate custom tracts.
mkdir -p $src/$output_folder/$output_tractography/$fba_output
mkdir -p $src/$output_folder/$output_tractography/$individual_tracts
cd $src/$output_folder/$output_tractography/$individual_tracts

for tract in ${tract_to_examine[@]};do
    mkdir -p $tract
done
run 'regriding KANA mask' \
	mrgrid IN:${KANA_in_template_space} regrid -voxel 1.3 -interp nearest OUT:$src/$output_folder/$output_5TT/$regrid_KANA_in_template

run 'Generating Interhemispheric exclusion regions'\
	mrconvert IN:$src/$output_folder/$warped_mask_average OUT:$interhemispheric_exclude --coord 0 37

run 'registering common average wm_fod to WM FOD parcellation space' \
	mrregister IN:$src/$output_folder/$warped_wm_fod_average IN:$common_wm_fod_40weeks_by_Alena -mask1 IN:$src/$output_folder/$warped_mask_average -nl_warp OUT:$src/$output_folder/$output_5TT/$native2wm_parc_warp OUT:$src/$output_folder/$output_5TT/$wm_parc2native_warp

run 'transforming wm parcellation to average space' \
	mrtransform IN:$wm_parcellation_by_Alena -warp IN:$src/$output_folder/$output_5TT/$wm_parc2native_warp -interp nearest OUT:$src/$output_folder/$output_5TT/$wm_parcellation_warped_wm_fod

#run 'Generating forceps minor' \
#    generate_wm_tract IN:$src/$wm_tracts CC \
#    -tckdo tckgen \
#    -seed_image IN:$fmi/genu_cc_mask.mif \
#    -fod IN:$src/$output_folder/$warped_wm_fod_average \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -mask IN:$src/$output_folder/$warped_mask_average \
#    -seeds 1000000 \
#    -select 0 \
#    -cutoff 0.1 \
#    -out OUT:$fmi/$fmi.tck

#run 'Generating forceps major' \
#    generate_wm_tract IN:$src/$wm_tracts CC_fma \
#    -tckdo tckgen \
#    -include IN:$fma/splenium_cc_mask.mif \
#    -ROI IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template \
#    -fod IN:$src/$output_folder/$warped_wm_fod_average \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -mask IN:$src/$output_folder/$warped_mask_average \
#    -select 0 \
#    -seeds 5000000 \
#    -angle 20 \
#    -cutoff 0.08 \
#    -out OUT:$fma/$fma.tck

#run 'Generating right posterior limb internal capsule' \
#    generate_wm_tract IN:$src/$wm_tracts PLIC_R \
#    -tckdo tckgen \
#    -exclude IN:$interhemispheric_exclude \
#    -ROI IN:$src/$output_folder/$output_5TT/$wm_parcellation_warped_wm_fod \
#    -fod IN:$src/$output_folder/$warped_wm_fod_average \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -mask IN:$src/$output_folder/$warped_mask_average \
#    -seed_image IN:$src/$output_folder/$output_5TT/thalamus_R_mask.mif \
#    -select 0 \
#    -seeds 50000000 \
#    -angle 20 \
#    -cutoff 0.2 \
#    -maxlength 50 \
#    -out OUT:$plic_R/$plic_R.tck \
#    -seed_unidirectional
#
#run 'Generating right posterior limb internal capsule' \
#    generate_wm_tract IN:$src/$wm_tracts PLIC_L \
#    -tckdo tckgen \
#    -exclude IN:$interhemispheric_exclude \
#    -ROI IN:$src/$output_folder/$output_5TT/$wm_parcellation_warped_wm_fod \
#    -fod IN:$src/$output_folder/$warped_wm_fod_average \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -mask IN:$src/$output_folder/$warped_mask_average \
#    -seed_image IN:$src/$output_folder/$output_5TT/thalamus_L_mask.mif \
#    -select 0 \
#    -seeds 50000000 \
#    -angle 20 \
#    -cutoff 0.2 \
#    -maxlength 50 \
#    -out OUT:$plic_L/$plic_L.tck \
#    -seed_unidirectional

#run 'Generating corticospinal tract L' \
#    generate_wm_tract IN:$src/$wm_tracts CST_L \
#    -tckdo tckgen \
#    -exclude IN:$interhemispheric_exclude \
#    -ROI IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template \
#    -fod IN:$src/$output_folder/$warped_wm_fod_average \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -mask IN:$src/$output_folder/$warped_mask_average \
#    -select 0 \
#    -seeds 10000000 \
#    -angle 20 \
#    -cutoff 0.2 \
#    -out OUT:$cst_L/$cst_L.tck \
#    -seed_unidirectional
#
#run 'Generating corticospinal tract R' \
#    generate_wm_tract IN:$src/$wm_tracts CST_R \
#    -tckdo tckgen \
#    -exclude IN:$interhemispheric_exclude \
#    -ROI IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template \
#    -fod IN:$src/$output_folder/$warped_wm_fod_average \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -mask IN:$src/$output_folder/$warped_mask_average \
#    -select 0 \
#    -seeds 10000000 \
#    -angle 20 \
#    -cutoff 0.2 \
#    -out OUT:$cst_R/$cst_R.tck \
#    -seed_unidirectional
#


#run 'Generating CC' \
#    generate_wm_tract IN:$src/$wm_tracts CC \
#    -tckdo tckedit \
#    -tckfile $src/$output_folder/$output_tractography/$reduced_tracts \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -mask IN:$src/$output_folder/$warped_mask_average \
#    -maxlength 10 \
#    -out OUT:$cc/$cc.tck
#run 'Generating Cingulum Dorsal Right tract' \
#    generate_wm_tract IN:$src/$wm_tracts CING_D_R \
#    -tckdo tckedit \
#    -ROI IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template\
#    -fod IN:$src/$output_folder/$warped_wm_fod_average \
#    -tckfile IN:CING_D_Left_10000000.tck \
#    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
#    -select 10000 \
#    -maxlength 80 \
#    -exclude IN:$interhemispheric_exclude \
#    -out OUT:test.tck -keep

#SIFT - on the whole brain tractography
#run 'Reduce the tractogram using SIFT (defined termination point)' \
#	tcksift IN:$src/$output_folder/$output_tractography/$tracts \
#	IN:$src/$output_folder/$warped_wm_fod_average \
#	-act IN:$src/$output_folder/$output_5TT/$image_5TT \
#	-term_num $tracts_streamlines \
#	OUT:$src/$output_folder/$output_tractography/$sift_tracts
#

 
for tract in ${tract_to_examine[@]};do
update_folder_if_needed run "'generating fixel-fixel connectivity'" \
  fixelconnectivity IN:$src/$output_folder/$fixel_mask IN:$tract/$tract.tck OUT:$src/$output_folder/$output_tractography/$fba_output/${tract}_${ffixel_matrix}
done

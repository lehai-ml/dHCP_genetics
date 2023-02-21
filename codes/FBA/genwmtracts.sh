#!/bin/bash

#This file is used to generate custom tracts.

mkdir -p $src/$output_folder/$output_tractography/$individual_tracts
cd $src/$output_folder/$output_tractography/$individual_tracts

for tract in ${tract_to_examine[@]};do
    mkdir -p $tract
done
run 'regriding KANA mask' \
	mrgrid IN:${KANA_in_template_space} regrid -voxel 1.3 -interp nearest OUT:$src/$output_folder/$output_5TT/$regrid_KANA_in_template

run 'Generating Interhemispheric exclusion regions'\
	mrconvert IN:$src/$output_folder/$warped_mask_average OUT:$interhemispheric_exclude --coord 0 37

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

run 'Generating CC' \
    generate_wm_tract IN:$src/$wm_tracts CC \
    -tckdo tckedit \
    -tckfile $src/$output_folder/$output_tractography/$reduced_tracts \
    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
    -mask IN:$src/$output_folder/$warped_mask_average \
    -maxlength 10 \
    -out OUT:$cc/$cc.tck
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


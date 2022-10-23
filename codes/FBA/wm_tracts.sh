#!/bin/bash


mkdir -p $output_folder/$output_5TT/$tract_binary_masks
cd $output_folder/$output_5TT/$tract_binary_masks

run 'Generating Interhemispheric exclusion regions'\
    mrconvert IN:$src/$output_folder/$warped_mask_average OUT:$interhemispheric_exclude --coord 0 37

run 'Generating UF Left include regions' \
    generate_binary_mask IN:$src/$wm_tracts "Include_UF_left:" IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$UF_Left_include

run 'Generating UF Left exclude regions' \
    generate_binary_mask IN:$src/$wm_tracts "Exclude_UF_left:" IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$UF_Left_exclude

number_of_streamlines=1000000
run 'UF left streamlines' \
    tckgen IN:$src/$output_folder/$warped_wm_fod_average \
    -seed_image IN:$src/$output_folder/$warped_mask_average \
    -mask IN:$src/$output_folder/$warped_mask_average \
    -act IN:$src/$output_folder/$output_5TT/$image_5TT \
    -select 0 \
    -seeds $number_of_streamlines \
    -include IN:$UF_Left_include \
    -exclude IN:$UF_Left_exclude \
    -exclude IN:$interhemispheric_exclude \
    OUT:$src/$output_folder/$output_tractography/$UF_Left_tract

#length selection
run 'UF left length selection' \
    tcksift IN:

#!/bin/bash
mkdir -p $src/$output_folder/$output_tractography/$individual_tracts
cd $src/$output_folder/$output_tractography/$individual_tracts

run 'regriding KANA mask' \
	mrgrid IN:${KANA_in_template_space} regrid -voxel 1.3 -interp nearest OUT:$src/$output_folder/$output_5TT/$regrid_KANA_in_template

# UF

run 'Generate UF_left seed regions' \
	generate_binary_mask IN:$src/$wm_tracts 'seed_image_UF_Left:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$seed_image_UF_Left

run 'Generate UF_right seed regions' \
	generate_binary_mask IN:$src/$wm_tracts 'seed_image_UF_Right:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$seed_image_UF_Right

run 'Generate Fronto middle/inferior orbital region (9+15: UF left)' \
	generate_binary_mask IN:$src/$wm_tracts 'UF_L_9_15:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_UF_9_15

run 'Generate Fronto middle/inferior orbital region (10+16: UF right)' \
	generate_binary_mask IN:$src/$wm_tracts 'UF_R_10_16:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_UF_10_16

run 'Generate Middle temporal region (83+87: UF left)' \
	generate_binary_mask IN:$src/$wm_tracts 'UF_L_83_87:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_UF_83_87

run 'Generate Middle temporal region (84+88: UF right)' \
	generate_binary_mask IN:$src/$wm_tracts 'UF_R_84_88:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_UF_84_88

run 'Generate Thalamus and Middle temporal region (77+85: UF left)' \
	generate_binary_mask IN:$src/$wm_tracts 'UF_L_77_85:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$exclude_image_UF_77_85

run 'Generate Thalamus and Middle temporal region (78+86: UF right)' \
	generate_binary_mask IN:$src/$wm_tracts 'UF_R_78_86:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$exclude_image_UF_78_86

#Cingulum Dorsal (31+33+35)
run 'Generate Cingulum Dorsal Left seed region' \
	generate_binary_mask IN:$src/$wm_tracts 'seed_image_CING_D_Left:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$seed_image_CING_D_Left

run 'Generate Cingulum Dorsal Right seed region' \
	generate_binary_mask IN:$src/$wm_tracts 'seed_image_CING_D_Right:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$seed_image_CING_D_Right

run 'Generate Cingulum Anterior Left region (31)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_L_31:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_CING_31

run 'Generate Cingulum Middle Left region (33)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_L_33:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_CING_33

run 'Generate Cingulum Posterior Left region (35)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_L_35:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_CING_35

run 'Generate Parahippocampal Left region (39)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_L_39:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$exclude_image_CING_39

run 'Generate Cingulum Anterior Right region (32)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_R_32:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_CING_32

run 'Generate Cingulum Middle Right region (34)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_R_34:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_CING_34

run 'Generate Cingulum Posterior Right region (36)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_R_36:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$include_image_CING_36

run 'Generate Parahippocampal Right region (40)' \
	generate_binary_mask IN:$src/$wm_tracts 'CING_R_40:' IN:$src/$output_folder/$output_5TT/$regrid_KANA_in_template OUT:$exclude_image_CING_40

run 'Generating Interhemispheric exclusion regions'\
	mrconvert IN:$src/$output_folder/$warped_mask_average OUT:$interhemispheric_exclude --coord 0 37

run 'Generate UF Left' \
	tckgen $src/$output_folder/$warped_wm_fod_average\
	-select $tracks_streamlines \
	-act $src/$output_folder/$output_5TT/$image_5TT \
	-mask $src/$output_folder/$warped_mask_average \
	-exclude $interhemispheric_exclude \
	-cutoff 0.08 \
	-maxlength 80 \
	-seed_image $seed_image_UF_Left\
	-include $include_image_UF_9_15 \
	-include $include_image_UF_83_87\
	-exclude $exclude_image_UF_77_85 \
	OUT:$UF_Left_tract

run 'Generate UF Right' \
	tckgen $src/$output_folder/$warped_wm_fod_average\
	-select $tracks_streamlines \
	-act $src/$output_folder/$output_5TT/$image_5TT \
	-mask $src/$output_folder/$warped_mask_average \
	-exclude $interhemispheric_exclude \
	-cutoff 0.08 \
	-maxlength 80 \
	-seed_image $seed_image_UF_Right\
	-include $include_image_UF_10_16\
	-include $include_image_UF_84_88\
	-exclude $exclude_image_UF_78_86 \
	OUT:$UF_Right_tract

run 'Generate CING Dorsal Left' \
	tckgen $src/$output_folder/$warped_wm_fod_average\
	-select $tracks_streamlines \
	-act $src/$output_folder/$output_5TT/$image_5TT \
	-mask $src/$output_folder/$warped_mask_average \
	-exclude $interhemispheric_exclude \
	-cutoff 0.065 \
	-maxlength 100 \
	-seed_image $seed_image_CING_D_Left\
	-include $include_image_CING_31\
	-include $include_image_CING_33\
	-include $include_image_CING_35\
	-exclude $exclude_image_CING_39 \
	OUT:$CING_D_Left_tract

run 'Generate CING Dorsal Right' \
	tckgen $src/$output_folder/$warped_wm_fod_average\
	-select $tracks_streamlines \
	-act $src/$output_folder/$output_5TT/$image_5TT \
	-mask $src/$output_folder/$warped_mask_average \
	-exclude $interhemispheric_exclude \
	-cutoff 0.065 \
	-maxlength 100 \
	-seed_image $seed_image_CING_D_Right\
	-include $include_image_CING_32\
	-include $include_image_CING_34\
	-include $include_image_CING_36\
	-exclude $exclude_image_CING_40 \
	OUT:$CING_D_Right_tract

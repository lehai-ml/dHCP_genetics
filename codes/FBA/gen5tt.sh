#!/bin/bash

#5TT generation: suitable for use with ACT in tckgen, using 40 week probabilistic template.
#and power in l=2 SH band in FOD template to recover internal capusle

echo '###################################'
echo '    'Generating 5TT
echo '###################################'

cd $src
mkdir -p ${output_folder}/${output_5TT}

cd ${output_folder}/${output_5TT}

#run 'regriding KANA mask' \
#  mrgrid IN:${KANA_in_template_space} regrid -voxel 1.3 -interp nearest OUT:$regrid_KANA_in_template
#
#run 'generating deep-gray matter KANA mask in template space' \
#  mrcalc IN:${regrid_KANA_in_template} 79 -lt IN:${regrid_KANA_in_template} -multiply 70 -gt OUT:$KANA_DGM
#
#run 'regriding tissue segmented (by draw E-M) image' \
#  mrgrid IN:${Tissue_segmented} regrid -voxel 1.3 -interp nearest OUT:$regrid_Tissue_segmented
#
#run 'generating grey matter mask' \
#  mrcalc IN:$regrid_Tissue_segmented 2 -eq 0 -gt OUT:$gm
#
#run 'generating deep gray matter mask' \
#  mrcalc IN:$regrid_Tissue_segmented 9 -eq IN:$KANA_DGM -add OUT:$dgm
#
## {[(tissue==7)+(tissue==8)> 0] - (KANA_DGM) + (tissue = 6)} > 0
#run 'generating white matter mask' \
#  mrcalc IN:$regrid_Tissue_segmented 7 -eq IN:$regrid_Tissue_segmented 8 -eq -add 0 -gt - \| mrcalc - IN:$KANA_DGM -neg -add IN:$regrid_Tissue_segmented 6 -eq -add 0 -gt - \| mrcalc IN:$regrid_Tissue_segmented 3 -eq $dgm -neg -add - -add 0 -gt OUT:$wm
#
#run 'generating CSF mask' \
#  mrcalc IN:$regrid_Tissue_segmented 1 -eq IN:$regrid_Tissue_segmented 5 -eq -add 0 -gt OUT:$csf
#
#run 'generating 5TT' \
#  mrcalc IN:$regrid_Tissue_segmented 0 -lt - \| mrcat IN:$gm IN:$dgm IN:$wm IN:$csf - OUT:$image_5TT
#
#running streamline seeding? 5tt2gwwmi?

#echo "Generating 5TT maps"

run 'Generating GM template' \
	mrconvert IN:$Tissue_segmented_probseg -coord 3 0 -axes 0,1,2 OUT:$gm

run 'Generating Internal capsule from FOD template' \
	mrgrid IN:$src/${output_folder}/$warped_wm_fod_average regrid -template IN:$gm - \| \
       	sh2power - -spectrum - \| \
	mrconvert - -coord 3 1 - \| \
	mrcalc - 0.0025 -sub 500 -mult -tanh 1 -add 2 -div OUT:ic_cutout.mif 

run 'Generating whole brain mask with missing hippocampus' \
	mrmath IN:$Tissue_segmented_probseg sum -axis 3 - \| mredit - -sphere 130,120,75 15 100 -sphere 65,120,75 15 100 OUT:wb.mif

run 'Generating hippocampus mask' \
	mrmath IN:$Tissue_segmented_probseg sum -axis 3 - \| \
	mrcalc IN:wb.mif - -sub OUT:hipp.mif

run 'Extracting DGM without hippocampus' \
	mrconvert IN:$Tissue_segmented_probseg -coord 3 6 -axes 0,1,2 OUT:dgm_wout_hipp.mif
#WM = DGM without the IC
run 'Extracting WM' \
	mrconvert IN:$Tissue_segmented_probseg -coord 3 2,5,7 - \| \
       	mrmath - sum -axis 3 OUT:wm_wout_ic.mif

run 'Generating DGM template map' \
	mrcalc IN:dgm_wout_hipp.mif 1 IN:ic_cutout.mif -sub -mult - \| \
	mrmath - IN:hipp.mif sum OUT:$dgm

run 'Generating WM template map' \
	mrcalc IN:dgm_wout_hipp.mif ic_cutout.mif -mult - \| \
	mrmath IN:wm_wout_ic.mif - sum OUT:$wm

run 'Generating CSF template map' \
	mrconvert IN:$Tissue_segmented_probseg -coord 3 0,3,4 - \| \
	mrmath - sum -axis 3 OUT:$csf

run 'Generating pathology map' \
	mrcalc IN:$gm 0 -mult OUT:$path

run 'Generating 5TT file' \
	mrcat IN:$gm IN:$dgm IN:$wm IN:$csf IN:$path -axis 3 - \| mrcalc - 100 -div - \| mrgrid - regrid -template $src/$output_folder/$warped_wm_fod_average OUT:$image_5TT



#!/bin/bash

#5TT generation

echo '###################################'
echo '    'Generating 5TT
echo '###################################'

cd $src
mkdir -p ${output_folder}/${output_5TT}

cd ${output_folder}/${output_5TT}

run 'regriding KANA mask' \
  mrgrid IN:${KANA_in_template_space} regrid -voxel 1.3 -interp nearest OUT:$regrid_KANA_in_template

run 'generating deep-gray matter KANA mask in template space' \
  mrcalc IN:${regrid_KANA_in_template} 79 -lt IN:${regrid_KANA_in_template} -multiply 70 -gt OUT:$KANA_DGM

run 'regriding tissue segmented (by draw E-M) image' \
  mrgrid IN:${Tissue_segmented} regrid -voxel 1.3 -interp nearest OUT:$regrid_Tissue_segmented

run 'generating grey matter mask' \
  mrcalc IN:$regrid_Tissue_segmented 2 -eq 0 -gt OUT:$gm

run 'generating deep gray matter mask' \
  mrcalc IN:$regrid_Tissue_segmented 9 -eq IN:$KANA_DGM -add OUT:$dgm

# {[(tissue==7)+(tissue==8)> 0] - (KANA_DGM) + (tissue = 6)} > 0
run 'generating white matter mask' \
  mrcalc IN:$regrid_Tissue_segmented 7 -eq IN:$regrid_Tissue_segmented 8 -eq -add 0 -gt - \| mrcalc - IN:$KANA_DGM -neg -add IN:$regrid_Tissue_segmented 6 -eq -add 0 -gt - \| mrcalc IN:$regrid_Tissue_segmented 3 -eq $dgm -neg -add - -add 0 -gt OUT:$wm

run 'generating CSF mask' \
  mrcalc IN:$regrid_Tissue_segmented 1 -eq IN:$regrid_Tissue_segmented 5 -eq -add 0 -gt OUT:$csf

run 'generating 5TT' \
  mrcalc IN:$regrid_Tissue_segmented 0 -lt - \| mrcat IN:$gm IN:$dgm IN:$wm IN:$csf - OUT:$image_5TT

#running streamline seeding? 5tt2gwwmi?


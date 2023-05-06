#!/bin/bash

mkdir -p $output_folder/$glass_brain_folder

cd $output_folder/$glass_brain_folder

echo '#######################'
echo 'Generating Glass brain'
echo '#######################'

run 'creating mask from average template mask' \
	mrgrid IN:$src/$output_folder/$warped_mask_average regrid -voxel 0.5 - \| \
	mrfilter - smooth -stdev 2 - \| \
	mrthreshold - -abs 0.5 OUT:$mask_threshold

run 'creating glass brain' \
	maskfilter IN:$mask_threshold dilate -npass 2 - \| mrcalc - IN:$mask_threshold -sub - \| maskfilter - dilate OUT:$glass_brain

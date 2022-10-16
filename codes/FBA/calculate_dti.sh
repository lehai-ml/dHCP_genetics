#!/bin/bash


for ID in ${ID_list[@]}; do
(

echo '###################################'
echo '    '$ID
echo '###################################'

cd $output_folder/$ID
run 'extracting DTI' \
  dwiextract IN:$src/$dwi_data/$ID/$dwi -shells 0,1000 - \| dwi2tensor -mask IN:$mask - OUT:$diffusion_tensor

run 'calculating FA' \
  tensor2metric IN:$diffusion_tensor -fa OUT:$dt_fa

run 'calculating ADC' \
  tensor2metric IN:$diffusion_tensor -adc OUT:$dt_adc

run 'calculating RD' \
  tensor2metric IN:$diffusion_tensor -rd OUT:$dt_rd

run 'calculating AD' \
  tensor2metric IN:$diffusion_tensor -ad OUT:$dt_ad

)||continue

done


#!/bin/bash

#run 'generate all available ID list' \
#  python generate_ID_list.py generate --folder IN:$rf_data --pattern sub*/ses* --duplicates \| python generate_ID_list.py generate --folder IN:$dwi_data --pattern sub*/ses* --duplicates \| python generate_ID_list.py generate --file IN:$participants_info --idcolumns 0 2 --prefix sub- ses- --apcolumns 5 3 4 --no-duplicates --out OUT:$all_available_IDs
#
#run 'selecting usable ID' \
#  python generate_ID_list.py select --file IN:$all_available_IDs --criteria 3 37 '\>=' 2 37 '\<' --group preterm \|  python generate_ID_list.py select --file IN:$all_available_IDs --criteria 3 37 '\>=' 2 37 '\>=' --group term --remove CC00666XX15 CC00702BN09 CC00526XX15 --out OUT:$usable_subjects
#
#if [ ! -f subjects_list.txt ]; then
#    cat $usable_subjects | grep ,term | shuf -n 125  > subjects_list.txt
#    cat $usable_subjects | grep ,preterm >> subjects_list.txt
#fi
#
run 'generate all available ID list' \
  python generate_ID_list.py generate --folder IN:$rf_data --pattern "sub*/ses*" --duplicates \| \
  python generate_ID_list.py generate --folder IN:$dwi_data --pattern "sub*/ses*" --duplicates \| \
  python generate_ID_list.py generate \
  --file IN:$euro_SCZ_PRS_term \
  --header 0 \
  --idcolumns ID Session_vol \
  --prefix sub- ses- \
  --apcolumns GA_vol PMA_vol Gender PRS_1e-08 PRS_1e-07 PRS_1e-06 PRS_1e-05 \
  PRS_0.0001 PRS_0.001 PRS_0.01 PRS_0.05 PRS_0.1 PRS_0.5 PRS_1 \
  euro_Anc_PC1 euro_Anc_PC2 euro_Anc_PC3\
  --no-duplicates \
  --out OUT:$all_available_IDs

#run 'selecting usable ID' \
#  python generate_ID_list.py select --file IN:$all_available_IDs --criteria 3 37 '\>=' 2 37 '\<' --group preterm \|  python generate_ID_list.py select --file IN:$all_available_IDs --criteria 3 37 '\>=' 2 37 '\>=' --group term --remove CC00666XX15 CC00702BN09 CC00526XX15 --out OUT:$usable_subjects
#
#if [ ! -f subjects_list.txt ]; then
#    cat $usable_subjects | grep ,term | shuf -n 125  > subjects_list.txt
#    cat $usable_subjects | grep ,preterm >> subjects_list.txt
#fi
#



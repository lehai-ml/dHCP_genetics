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
  --idcolumns ID Session \
  --prefix sub- ses- \
  --apcolumns GA PMA sex \
  SCZ_PRS_Pt_1e-08 SCZ_PRS_Pt_1e-07 SCZ_PRS_Pt_1e-06 SCZ_PRS_Pt_1e-05 SCZ_PRS_Pt_0.0001 SCZ_PRS_Pt_0.001 SCZ_PRS_Pt_0.01 SCZ_PRS_Pt_0.05 SCZ_PRS_Pt_0.1 SCZ_PRS_Pt_0.5 SCZ_PRS_Pt_1 \
  ASD_PRS_Pt_1e-08 ASD_PRS_Pt_1e-07 ASD_PRS_Pt_1e-06 ASD_PRS_Pt_1e-05 ASD_PRS_Pt_0.0001 ASD_PRS_Pt_0.001 ASD_PRS_Pt_0.01 ASD_PRS_Pt_0.05 ASD_PRS_Pt_0.1 ASD_PRS_Pt_0.5 ASD_PRS_Pt_1 \
  euro_Anc_PC1 euro_Anc_PC2 euro_Anc_PC3 euro_Anc_PC4 euro_Anc_PC5\
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



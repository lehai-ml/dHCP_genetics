#!/bin/bash

dwi_data="data"
rf_data="dhcp_neo_dMRI_derived"
participants_info="dHCP_participant_info.csv"
usable_subjects="usable_subj.txt"

all_available_IDs=all_available_IDs.txt
usable_subj=usable_subj.txt

run 'generate all available ID list' \
  python generate_ID_list.py generate --folder IN:$rf_data --pattern sub*/ses* \| python generate_ID_list.py generate --folder IN:$dwi_data --pattern sub*/ses* \| python generate_ID_list.py generate --file IN:$participants_info --columns 0 2 --prefix sub- ses- --apcolumns 5 3 4 --out OUT:$all_available_IDs

run 'selecting usable ID' \
  python generate_ID_list.py select --file IN:$all_available_IDs --criteria 3 37 '\>=' 2 37 '\<' --group preterm \|  python generate_ID_list.py select --file IN:$all_available_IDs --criteria 3 37 '\>=' 2 37 '\>=' --group term --out OUT:$usable_subjects

if [ ! -f subjects_list.txt ]; then
    shuf -n 2 $usable_subjects > subjects_list.txt
fi



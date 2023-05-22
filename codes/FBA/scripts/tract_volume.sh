#!/bin/bash
#this is for calculating the tract volume using Alena's parcellation.
#The parcellation is created in perform_aba.sh

function calculate_volumes() {
	subject_list=$1
	output_file=$2
	output_file=$(echo ${output_file/tmp-})
	ID_list=()
	output_ID_list=()
	while read subj; do
		if [ "x$subj" == "x" ]; then continue; fi
		if [[ "$subj" =~ ^[[:space:]]*# ]]; then continue; fi
		IFS=',' read -ra subj <<< $subj
		ID_list+=(${subj[0]})
	done < $subject_list
	if [[ -f $output_file ]]; then
	
		while read subj; do
			if [ "x$subj" == "x" ]; then continue; fi
			IFS=' ' read -ra subj <<< $subj
			output_ID_list+=(${subj[0]})
		done < $output_file
	fi
	for ID in ${ID_list[@]}; do
		if [[ " ${output_ID_list[*]} " == *" $ID "* ]]; then
		       continue
		fi
 		
		echo calculating $ID
		volumes=()
		mrconvert $output_folder/$ID/$subject_wm_parc $output_folder/$ID/tmp-wm_in_subject_space.nii.gz
		segm_file=$output_folder/$ID/tmp-wm_in_subject_space.nii.gz
		volume=$(fslstats -K $segm_file $segm_file -V | awk '{print $2}')
		volumes+=($volume)
		echo $ID ${volumes[@]} >> $output_file
		rm $segm_file

done
}

calculate_volumes $subjects_list $output_folder/$aba/$tract_volumes 

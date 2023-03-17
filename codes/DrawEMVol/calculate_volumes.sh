#!/bin/bash

#Relevant files to access
src="$(pwd)"

volume_data=neonatal_release3
segmentation_87=_desc-drawem87_dseg.nii.gz
segmentation_9=_desc-drawem9_dseg.nii.gz

subjects_list=missing_ID.txt
volume_output=volumetric_DrawEM.txt
tissue_seg_output=9TT.txt

set -e
. ../FBA/support_functions.sh

if [ ! -d $volume_data ]; then
	echo "neonatal release not yet here"
	sudo mount -t cifs //isi01/dhcp-pipeline-data /home/lh20/dhcp-pipeline-data/ -o username=lh20,domain=isd,iocharset=utf8,vers=2.1

fi


cd $src

function calculate_volumes() {
	subject_list=$1
	segmentation=$2
	output_file=$3
	output_file=$(echo ${output_file/tmp-})
	ID_list=()
	output_ID_list=()
	while read subj; do
		if [ "x$subj" == "x" ]; then continue; fi

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
		(
		if [[ " ${output_ID_list[*]} " == *" $ID "* ]]; then
			continue
		fi
		
		echo calculating $ID
		id_ses=$(echo $ID | sed 's/\//_/')
		segm_file=$volume_data/$ID/anat/${id_ses}${segmentation}
		volumes=()
		volume=$(fslstats -K $segm_file $segm_file -V | awk '{print $2}')
		volumes+=($volume)
		echo $ID ${volumes[@]} >> $output_file
		)
	done
}


run 'calculating volumes' calculate_volumes IN:$subjects_list $segmentation_87 OUT:$volume_output
run 'calculating 9 TT volumes' calculate_volumes IN:$subjects_list $segmentation_9 OUT:$tissue_seg_output




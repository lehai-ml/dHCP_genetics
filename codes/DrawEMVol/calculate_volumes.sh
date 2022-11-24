#!/bin/bash

#Relevant files to access
src="$(pwd)"

volume_data=neonatal_release3
segmentation=_desc-drawem87_dseg.nii.gz
subjects_list=subjects_list.txt
volume_output=volumetric_DrawEM.txt

set -e
. ../docs/bash/support_functions.sh

if [ ! -d $volume_data ]; then
	echo "neonatal release not yet here"
	sudo mount -t cifs //isi01/dhcp-pipeline-data /home/lh20/dhcp-pipeline-data/ -o username=lh20,domain=isd,iocharset=utf8,vers=2.1

fi


cd $src

function calculate_volumes() {
	subject_list=$1
	output_file=$2
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
		for value in `seq 87`; do
			lower_t=$(( $value - 1 ))
			upper_t=$(( $value + 1 ))
			volume=$(fslstats -t $segm_file -l $lower_t -u $upper_t -V | awk '{print $2}')
			volumes+=($volume)
		done
		echo $ID ${volumes[@]} >> $output_file
		)
	done
}


run 'calculating volumes' calculate_volumes IN:$subjects_list OUT:$volume_output





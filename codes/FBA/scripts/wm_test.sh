#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'
function generate_binary_ROI_mask {
    ROIs_template=$1
    IFS='+' read -r -a ROIs <<< $2
    output=$3
    to_eval="mrcalc "
    for (( i=0; i<${#ROIs[@]}; ++i ));do
	region="${ROIs[i]}"
	to_eval+=$(echo $ROIs_template $region -eq " ")
	if [ $i -gt 0 ];then
	    to_eval+="-max "
	fi
    done
    to_eval+=$output
    eval "${to_eval[@]}"
}
function generate_wm_tract {
	file=$1
	identifier=$2
	include=()
	while read line; do
	    if [[ $line == $(echo "$identifier*") ]];then
	        include+=(${line#$identifier})
	    fi
	done < $file
	cmd=($@)
	for i in "${!cmd[@]}";do
	    if [[ ${cmd[$i]} == "-ROI" ]]; then
		ROI="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-tckdo" ]]; then
		track_cmd="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-tckfile" ]]; then
		track="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-fod" ]]; then
		fod="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-act" ]]; then
		act="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-out" ]]; then
		output="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-select" ]]; then
		select="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-seeds" ]]; then
		seeds="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-mask" ]]; then
	        mask_tract="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-maxlength" ]]; then
	        maxlength="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-minlength" ]]; then
	        minlength="${cmd[(($i+1))]}"
	    elif [[ ${cmd[$i]} == "-fo" ]]; then
		fo="${cmd[$i]}"
	    elif [[ ${cmd[$i]} == "-keep" ]]; then
		keep=1
	    fi
	done
	
	to_eval=()
	to_eval+=$track_cmd
	if [[ $track_cmd == "tckgen" ]]; then
	    if [ -z "${fod+x}" ];then
		echo "FOD image is required to do tckgen iFOD2"
		return 0
	    fi
	    to_eval+=" $fod"
	    for n in "${!include[@]}"; do
		if [[ ${include[$n]} == "-seed_image:" ]]; then
		    seed_roi=${include[(($n+1))]}
		    echo -e "${RED}>>> generating seed image ${NC}"
		    seed_image="$identifier-tmp-seed_image.mif"
		    (generate_binary_ROI_mask $ROI $seed_roi $seed_image)
		    to_eval+=" -seed_image $seed_image"
		elif [[ ${include[$n]} == "-seed_sphere:" ]]; then
		    seed_sphere=${include[(($n+1))]}
		    to_eval+=" -seed_sphere $seed_sphere"
		fi
	    done
	    if [ ! -z "${act+x}" ];then
	        to_eval+=" -act $act"
	    fi
	    if [ ! -z "${seeds+x}" ];then
	        to_eval+=" -seeds $seeds"
	    fi
	    if [ ! -z "${select+x}" ];then
	        to_eval+=" -select $select"
	    fi
	elif [[ $track_cmd == "tckedit" ]]; then
	    if [ -z "${track+x}" ]; then
	        echo -e "${RED} tck file is needed to do tckedit ${NC}"
		return 0
	    fi
	    to_eval+=" $track"
	fi
	for n in "${!include[@]}";do
	    if [[ ${include[$n]} == "-include:" ]] || [[ ${include[$n]} == "-exclude:" ]] ; then
	        to_do=${include[$n]%:}
	        to_include=${include[(($n+1))]}
	        if [[ $to_include == *"+"* ]] || [ "$to_include" -eq "$to_include" ] ; then
	            if [ -z "${ROI+x}" ]; then
	                echo "ROI needed to create include/exclude mask"
			return 0
	            fi
		    region="$identifier-tmp$to_do-$n.mif"
		    echo 
	            (generate_binary_ROI_mask $ROI $to_include $region)
		    to_eval+=" $to_do $region"
		else
		    to_eval+=" $to_do $to_include"
	        fi
	    fi
	done
	if [ ! -z "${mask_tract+x}" ];then
	    to_eval+=" -mask $mask_tract"
	fi
	if [ ! -z "${maxlength+x}" ];then
	    to_eval+=" -maxlength $maxlength"
	fi
	if [ ! -z "${minlength+x}" ];then
	    to_eval+=" -minlength $minlength"
	fi
	to_eval+=" $output $fo"
	echo "${to_eval[@]}"
	eval "${to_eval[@]}"
	if [ -z "${keep+x}" ];then
	    rm $identifier-tmp-*.mif 2>/dev/null
	fi
}

#. support_functions.sh

#run 'do something' \
#	generate_wm_tract IN:wm_tracts.txt CING_D_R -tckdo tckgen -ROI output/5TT/regrid_KANA_in_template_space.mif -tckdo tckgen -fod output/warped_wm_fod_average.mif -select 1000 -out OUT:test.tck

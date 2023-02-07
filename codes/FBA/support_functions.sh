
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Convenience function to check whether outputs already exist, and 
# whether inputs are newer than outputs, in which case the processing 
# should be run again. 
#
# Usage: 
#   needs_updating output_file [output_file2 ...] -- input_file [input_file2 ...]
#
# Returns 0 if need to recompute, 1 otherwise
function needs_updating () {
  output=$1
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  NC='\033[0m'
  while [ $# -gt 0 ]; do
    if [ $1 == "--" ]; then break; fi
    if [ ! -e $1 -a ! -e ../$1 -a ! -e ../../$1 -a ! -e ../../../$1 -a ! -e $output_dir/$1 ]; then
      # the file may be found in another subfolder (e.g., this happens when using fod2fixel -afd or fixelreorientation)
      echo -e "${RED} outputs missing - recomputing${NC}"
      return 1;
    else
      if [ -d $1 ]; then # if the output is a directory that exist
         output_dir=$1
      fi
    fi
    shift
  done
  shift
  while [ $# -gt 0 ]; do
    if [ $1 -nt $output ]; then
      echo -e "${RED} inputs have been updated - recomputing${NC}"
      return 1
    fi
    shift
  done
  echo -e "${GREEN} already up to date - skipping${NC}"
  return 0
}



# Convenience function to run a step in the preprocessing if
# outputs are missing or inputs are newer than outputs. 
#
# usage:
#   run 'progress message' cmd [args...]
#
# where args can be prefixed with IN: (e.g. IN:image.nii) to denote an 
# input file, or prefixed with OUT: to denote an output
function run {
  echo -e -n "> ${GREEN}$1...${NC}"
  cmd=($2)
  shift 2

  inputs=()
  outputs=()

  for arg in "$@"; do
    if [[ $arg == IN:* ]]; then 
      arg=${arg#IN:}
      inputs+=($arg)
      cmd+=($arg)
    elif [[ $arg == OUT:* ]]; then
      arg=${arg#OUT:}
      outputs+=($arg)
      if [ -d $arg ]; then
	cmd+=($arg)
	continue
      fi
      if [[ $arg == */* ]];then
	arg=$(echo ${arg%\/*}"/tmp-"${arg##*\/})
	cmd+=($arg)
      else
	cmd+=(tmp-$arg)
      fi	      
    else
      cmd+=($arg)
    fi
  done

  #echo inputs=${inputs[@]}
  #echo outputs=${outputs[@]}
  #echo ${cmd[@]}
  (needs_updating ${outputs[@]} -- ${inputs[@]}) || 
  (
  eval ${cmd[@]}
  retval=$?
  if [ $retval -eq 0 ]; then
    for out in ${outputs[@]}; do
      if [ -d $out ]; then continue; fi; # if the folder already existed, continue
      if [[ $out == */* ]]; then # if it is a first-time folder, then append tmp- to the file.
        mv $(echo ${out%\/*}"/tmp-"${out##*\/}) $out
      else
	mv tmp-$out $out  2>/dev/null #sometimes the file is output to a folder, not the current one 
        if [ $? -eq 1 ]; then
          output_dir=$(find ../../ -name "tmp-${out}") # sometimes the files is output to the folder in parents directory
          mv $output_dir ${output_dir%/*}/$out 
        fi
      fi
    done
  fi

  rm -rf tmp-*

  return $retval
  )
}

function update_folder_if_needed() {

	(eval "${@}") ||
	(for arg in "$@"; do
	    if [[ $arg == OUT:* ]]; then
		arg=${arg#OUT:}
		if [ -d $arg ]; then
		   echo -e "${RED} Removing "$arg" folder${NC}"
		   rm -rf $arg
	        fi
	    fi
	done
	eval "${@}" )
}

# Convenience function to prefix all arguments with IN:
# This is useful when using pattern matching to capture all inputs
# when passing a command to the 'run' function above, using command substitution.
#
# For example:
#   run 'an example command aggregating a lot of input files' \
#     mrmath $(IN *.nii) mean OUT:mean.nii
function IN {
  for x in $@; do
    echo IN:$x
  done
}
#check if content in a folder match with content in a file
function sanity_check {
  echo -e "${GREEN}> checking if IDs match ...${NC}"
  pattern_to_match=$1
  dest1=$2
  shift 2
  if [ -d $dest1 ]; then
    file1=$(find ${dest1} -maxdepth 1 -name "${pattern_to_match}*" -exec basename \{} ${pattern_to_match} \;)
  elif [ -f $dest1 ]; then
    file1=$(cat $dest1)
  fi
  for dest2 in "$@"; do
   if [ -d $dest2 ]; then
    file2=$(find ${dest2} -maxdepth 1 -name "${pattern_to_match}*" -exec basename \{} ${pattern_to_match} \;)
  elif [ -f $dest2 ]; then
    file2=$(cat $dest2)
  fi
  difference=$(echo ${file1[@]} ${file2[@]} | tr ' ' '\n' | sort | uniq -u)
  if [[ ${#difference} -gt 0 ]]; then
    echo -e "${RED} ${dest1} is different from ${dest2} ${NC}"
    echo -e "${RED} removing ${dest2} ${NC}"
    rm -rf ${dest2}
  fi
  done
}

# Convenience function to generate mask of ROIs to be using mrcalc.
# For example, the result can be used in tckgen to include or exclude
# certain tracks.
# usage:
# generate_track_mask [file] [identifier] [binary_mask] [output]
# where file is a text file containing the
#    Include: 10 11 12 
#    Include: 13 14
#    Exclude: 15 16 17 18 19
#Where each rows that you want to combine is preceded by the same identifier.
#e.g. Include
#For example generate_track_mask file.txt "Include:" [binary_mask] [output]
# will reiteratively get regions 10, 11, 12, 13, 14 in the binary mask and
# combine into a single image using mrcalc.
#e.g.max(max(max((mrcalc binary_mask == 10), (binary_mask==11)),
#max((binary_mask==12),(binary_mask==13))),(binary_mask==14))

function generate_binary_mask {
    file=$1
    identifier=$2
    include=()
    while read line; do
    if [[ $line == $(echo $identifier*) ]]; then
	include+=(${line#$identifier})
    fi
    done < $file
    shift 2
    template=$1
    shift
    #create mask of things to include and exclude
    to_eval="mrcalc "
    output=$1
    for (( i=0; i<${#include[@]}; ++i )); do
	current_count=$((i+1))
        region="${include[i]}"
        to_eval+=$(echo $template $region -eq " ")
	while [[ $((current_count % 2 )) -eq 0 ]]; do
	    to_eval+="-max "
	    current_count=$((current_count/2))
	done
    done
    to_eval+=$output
    eval ${to_eval[@]}
}

#convenient function that copy header into a new file
function copy_header {
	file1=$1
	file2=$2
	shift 2
	folders=$@
	for folder in ${folders[@]}; do
	    for file in $file1 $file2; do head -n 1 $file >> $folder; done
	done
}


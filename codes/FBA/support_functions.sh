

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
  while [ $# -gt 0 ]; do
    if [ $1 == "--" ]; then break; fi
    if [ ! -e $1 ]; then
      echo ' outputs missing - recomputing'
      return 0;
    fi
    shift
  done
  shift
  while [ $# -gt 0 ]; do
    if [ $1 -nt $output ]; then
      echo " inputs have been updated - recomputing"
      return 0
    fi
    shift
  done
  echo ' already up to date - skipping'
  return 1
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
  echo -n "> $1..."
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
      cmd+=(tmp-$arg)
    else
      cmd+=($arg)
    fi
  done

  #echo inputs=${inputs[@]}
  #echo outputs=${outputs[@]}

  needs_updating ${outputs[@]} -- ${inputs[@]} || return 0

  eval ${cmd[@]}
  retval=$?
  if [ $retval -eq 0 ]; then
    for out in ${outputs[@]}; do
      mv tmp-$out $out
    done
  fi
  rm -f tmp-*

  return $retval
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



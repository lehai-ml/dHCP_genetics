#!/bin/bash


while getopts "hb:l:t:v:o:" arg; do
    case $arg in
	b) base_file=${OPTARG};;
	l) ld_file=${OPTARG};;
	t) target_file=${OPTARG};;
	v) valid_snp_file=${OPTARG};;
	o) output_folder=${OPTARG};;
	h) 
	    echo "-h: help menu"
	    echo "-b: path to summary statistics base file"
	    echo "-l: path to ld b-file"
	    echo "-t: path to target b-file"
	    echo "-v: path to valid extract SNP file, this is outputed by PRSice after the first run"
	    echo "-o: path to output file name root"
	    exit 0;;
	?)  exit 1;;
    esac
done

[[ -z "${base_file}" ]] && echo "ERROR: -b base file is missing" && exit 1
[[ -z  "${target_file}" ]] && echo "ERROR: -t target file is missing" && exit 1
[[ -z "${ld_file}" ]] && echo "ERROR: -l path to ld file is missing" &&  exit 1
[[ -z  "${output_folder}" ]] && echo "ERROR: -o path to and output filename root is missing" && exit 1

dir=~/Desktop/dHCP_genetics/codes/gene_set/PRSice # directory that holds the PRSice script and executable
PRSice_script=PRSice.R #path to the PRSice R script
PRSice_bin=PRSice_linux # path to the PRSice executable
#lower=1e-8
#upper=1
#number=100
#PRS_threshold=$(python generate_thresholds_intervals.py \
#	--lower $lower --upper $upper --log --number $number --precision 1)
#
#change the bar-levels to adjust. if you want an interview you can 
#uncomment the PRS_threshold=$(python...) and define the interval 

if [[ -z "${valid_snp_file}" ]]; then
Rscript $dir/$PRSice_script\
	--prsice $dir/$PRSice_bin \
	--dir $dir \
	--a1 A1\
	--a2 A2\
	--bar-levels 1e-8,1e-6,1e-5,0.0001,0.001,0.01,0.05,0.1,0.5,1\
	--fastscore \
	--base $base_file\
	--binary-target T\
	--bp BP\
	--chr CHR\
	--snp SNP\
	--stat OR\
	--target $target_file \
	--out $output_folder \
	--no-regress \
	--print-snp \
	--ld $ld_file
else
Rscript $dir/$PRSice_script\
	--prsice $dir/$PRSice_bin \
	--dir $dir \
	--a1 A1\
	--a2 A2\
	--bar-levels 1e-8,1e-6,1e-5,0.0001,0.001,0.01,0.05,0.1,0.5,1\
	--fastscore \
	--base $base_file\
	--binary-target T\
	--bp BP\
	--chr CHR\
	--snp SNP\
	--stat OR\
	--target $target_file \
	--out $output_folder \
	--no-regress \
	--print-snp \
	--ld $ld_file \
	--extract $valid_snp_file
fi

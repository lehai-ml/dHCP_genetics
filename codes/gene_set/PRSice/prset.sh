#!/bin/bash

src=$(pwd)
genetic_data=genetic_data
PRSice_script=PRSice.R
PRSice_bin=PRSice_linux
base_file=$src/$genetic_data/base_files/asd/iPSYCH-PGC_ASD_Nov2017
target_files=$src/$genetic_data/target_files/lifted37_dHCP_merged_cleaned_EUROPEANS #set of files containing .bed .bim .fam
gene_build=$src/$genetic_data/gene_build/NCBI37.3.gene.loc
msigdb=$src/$genetic_data/pathway_database/MSigDB/MSigDB_custom_entrez.gmt

. support_functions.sh

run 'performing PRSet' \
Rscript $PRSice_script \
	--prsice $PRSice_bin \
	--base IN:$base_file\
	--target IN:$target_files\
	--binary-target T\
	--thread 1\
	--gtf IN:$gene_build \
	--msigdb

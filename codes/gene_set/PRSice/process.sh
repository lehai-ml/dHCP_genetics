#!/bin/bash

src=$(pwd)
genetic_data=genetic_dataset
PRSice_script=PRSice.R
PRSice_bin=PRSice_linux

output_folder=output

#base_file=$src/$genetic_data/base_files/asd/preprocessed_iPSYCH-PGC_ASD_Nov2017.gz
base_file=$src/$genetic_data/base_files/scz/PGC3_SCZ_short.filtered_and_noambiguous_alt2.gz
target_files=$src/$genetic_data/target_files/euro_batch2_genotyped #set of files containing .bed .bim .fam
gene_build=$src/$genetic_data/$gene_build
#Ensemblegtf=Homo_sapiens.GRCh37.87.gtf
#msigdb=$src/$genetic_data/pathway_database/MSigDB/jansen_gene_set_symbol.gmt
ld_files=$src/$genetic_data/ld_files/PRSICE_check_file #set of files containing .bed .bim.fam
#pheno_cov_files=$src/$genetic_data/pheno_cov_files/asd
#phenotype=phenotype_EUR.txt
#covariate=covariate_EUR.txt
lower=1e-8
upper=1
number=100
PRS_threshold=$(python generate_thresholds_intervals.py \
	--lower $lower --upper $upper --log --number $number --precision 1)
#Rscript ./$PRSice_script\
#	--prsice ./$PRSice_bin \
#	--dir . \
#	--a1 A1\
#	--a2 A2\
#	--bar-levels 1e-8,1e-6,1e-5,0.0001,0.001,0.01,0.05,0.1,0.5,1\
#	--fastscore \
##	--all-score \
#	--base $base_file\
#	--base-info INFO:0.9\
#	--binary-target F\
#	--bp BP\
#	--chr CHR\
#	--snp SNP\
#	--stat OR\
##	--cov $pheno_cov_files/$covariate\
##	--pheno $pheno_cov_files/$phenotype\
##	--gtf $gene_build/$Ensemblegtf\
#	--ld $ld_files\
##	--msigdb $msigdb\

Rscript ./$PRSice_script\
	--prsice ./$PRSice_bin \
	--dir . \
	--a1 A1\
	--a2 A2\
	--bar-levels $PRS_threshold\
	--fastscore \
	--base $base_file\
	--base-info INFO:0.9\
	--binary-target F\
	--bp BP\
	--chr CHR\
	--snp SNP\
	--stat OR\
	--ld $ld_files\
	--target $target_files \
	--out $src/$output_folder/scz/PRS_100 \
	--no-regress \
	--extract $src/$output_folder/scz/PRS.valid


#!/bin/bash

src=$(pwd)
genetic_data=genetic_data
PRSice_script=PRSice.R
PRSice_bin=PRSice_linux
output_folder=output
base_file=$src/$genetic_data/base_files/asd/preprocessed_iPSYCH-PGC_ASD_Nov2017.gz
target_files=$src/$genetic_data/target_files/lifted37_dHCP_merged_cleaned_EUROPEANS #set of files containing .bed .bim .fam
gene_build=$src/$genetic_data/gene_build
Ensemblegtf=Homo_sapiens.GRCh37.87.gtf
msigdb=$src/$genetic_data/pathway_database/MSigDB/jansen_gene_set.gmt
ld_files=$src/$genetic_data/ld_files/PRSICE_check_file #set of files containing .bed .bim.fam
pheno_cov_files=$src/$genetic_data/pheno_cov_files/asd
phenotype=phenotype_EUR.txt
covariate=covariate_EUR.txt


. support_functions.sh
run 'attempting PRSet'\
	Rscript ./$PRSice_script\
	--prsice ./$PRSice_bin \
	--dir . \
	--a1 A1\
	--a2 A2\
	--bar-levels 1\
	--base IN:$base_file\
	--base-info INFO:0.9\
	--binary-target F\
	--bp BP\
	--chr CHR\
	--snp SNP\
	--stat OR\
	--clump-kb 1000kb\
	--clump-p 1\
	--clump-r2 0.1\
	--cov IN:$pheno_cov_files/$covariate\
	--pheno IN:$pheno_cov_files/$phenotype\
	--gtf IN:$gene_build/$Ensemblegtf\
	--ld IN:$ld_files\
	--msigdb IN:$msigdb\
	--target IN:$target_files \
	--or \
	--out OUT:$src/$output_folder/asd/test \
	--extract $src/$output_folder/asd/tmp-test.valid


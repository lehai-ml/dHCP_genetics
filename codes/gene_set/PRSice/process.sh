#!/bin/bash

src=$(pwd)
genetic_data=genetic_data
PRSice_script=PRSice.R
PRSice_bin=PRSice_linux
output_folder=output
base_file=$src/$genetic_data/base_files/asd/preprocessed_iPSYCH-PGC_ASD_Nov2017.gz
target_files=$src/$genetic_data/target_files/lifted37_dHCP_merged_cleaned_EUROPEANS_preprocessed #set of files containing .bed .bim .fam
gene_build=$src/$genetic_data/gene_build
Ensemblegtf=Homo_sapiens.GRCh37.87.gtf
msigdb=$src/$genetic_data/pathway_database/MSigDB/jansen_gene_set_symbol.gmt
ld_files=$src/$genetic_data/ld_files/PRSICE_check_file #set of files containing .bed .bim.fam
pheno_cov_files=$src/$genetic_data/pheno_cov_files/asd
phenotype=phenotype_EUR.txt
covariate=covariate_EUR.txt


Rscript ./$PRSice_script\
	--prsice ./$PRSice_bin \
	--dir . \
	--a1 A1\
	--a2 A2\
	--bar-levels 1e-8,1e-6,1e-5,0.0001,0.001,0.01,0.05,0.1,0.5,1\
	--fastscore \
	--all-score \
	--base $base_file\
	--base-info INFO:0.9\
	--binary-target F\
	--bp BP\
	--chr CHR\
	--snp SNP\
	--stat OR\
	--cov $pheno_cov_files/$covariate\
	--pheno $pheno_cov_files/$phenotype\
	--gtf $gene_build/$Ensemblegtf\
	--ld $ld_files\
	--msigdb $msigdb\
	--target $target_files \
	--out $src/$output_folder/asd/test \
	--no-regress \
	--extract $src/$output_folder/asd/tmp-test.valid


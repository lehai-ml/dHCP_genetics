#!/bin/bash

src=$(pwd)
genetic_data=genetic_dataset
PRSice_script=PRSice.R
PRSice_bin=PRSice_linux
#base_file=$src/$genetic_data/base_files/scz/PGC3_SCZ_european.filtered_and_noambiguous_alt2
base_file=$src/$genetic_data/base_files/asd/iPSYCH-PGC_ASD_Nov2017.info_filtered_and_noambiguous_alt2
target_files=$src/$genetic_data/target_files/batch2/euro_batch2_imputed #set of files containing .bed .bim .fam
gene_build=$src/$genetic_data/gene_build/Homo_sapiens.GRCh37.87.gtf
msigdb=$src/$genetic_data/pathway_database/MSigDB/MSigDB_custom_entrez.gmt
#msigdb=$src/$genetic_data/pathway_database/MSigDB/msigdb_gene_ontology.gmt
#msigdb=$src/all_pathways.gmt
#msigdb=$src/SCZ_enriched_pathways_AMIGO.gmt
ld_files=$src/$genetic_data/ld_files/PRSICE_check_file
covariates=covariates.txt
phenotypes=phenotypes.txt
output=ASD_PRS

mkdir -p $src/$output

Rscript $PRSice_script \
	--prsice $PRSice_bin \
	--bar-levels 1 \
	--fastscore \
	--binary-target F \
	--base $base_file\
	--target $target_files\
	--thread 1\
	--gtf $gene_build \
	--msigdb $msigdb \
	--ld $ld_files \
	--cov $covariates \
	--cov-col sex,GA,PMA,TBV \
	--cov-factor sex \
	--pheno $phenotypes \
	--pheno-col log_FC_SCR \
	--extract prset_SCZ.valid \
	--all-score \
	--set-perm 5000 \
	--no-regress \
	--out $scr/$output/log_FC_SCR

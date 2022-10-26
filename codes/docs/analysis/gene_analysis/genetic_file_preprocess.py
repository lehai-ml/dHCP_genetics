""""
genetic_file_preprocess.py
preprocess prsice, magma and plink file outputs
"""
from typing import List, Union, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from scipy import stats
import matplotlib.pyplot as plt
class Cohort:
    def __init__(self,
                 cohort_name:str=None,
                 PRS_file_path:str=None,
                 Ancestry_file_path:str=None,
                 imaging_df:pd.DataFrame=None):
        """_summary_

        Args:
            cohort_name (str, optional): _description_. Defaults to None.
            PRS_file_path (str, optional): _description_. Defaults to None.
            Ancestry_file_path (str, optional): _description_. Defaults to None.
            imaging_file_path (Optional[str,pd.DataFrame], optional): _description_. Defaults to None.
        """
        self.cohort_name = cohort_name
        self.prs = self.preprocess_PRSice_PRS_Anc_files(PRS_file_path)
        self.ancestry = self.preprocess_PRSice_PRS_Anc_files(Ancestry_file_path)
        self.prs.columns = ['PRS_' + i for i in self.prs.columns if i != 'ID']
        self.ancestry.columns = [f'{self.cohort_name}_Anc_' + i for i in self.ancestry.columns if 'PC' in i]
        self.genetic_df = pd.concat([self.prs,self.ancestry],join='inner',axis=1)
        self.imaging_df = imaging_df
        self.cohort_data = self.combine_imaging_genetic_data(self.imaging_df,self.genetic_df)
        self.cohort_data['termness'] = self.get_termness(self.cohort_data)
    
    def append_cohort_list(self,cohort_list:pd.DataFrame)-> pd.DataFrame:
        """
        Args:
            cohort_list (pd.DataFrame): index = IDs, cohort columns
        """
        self.cohort_data = self.cohort_data.join(cohort_list)
    
    @staticmethod
    def get_termness(df):
        return [
                'term' if
                (ga_struc >= 37) | (ga_micro >= 37) else 'preterm/term_at_scan' if
                ((ga_struc < 37) & (pma_struc >= 37)) |
                ((ga_micro < 37) & (pma_micro >= 37)) else 'preterm' if
                (ga_struc < 37) | (ga_micro < 37) else 'not available'
                for (ga_struc, pma_struc, ga_micro, pma_micro) in np.asarray(df[
                    ['GA_vol', 'PMA_vol', 'GA_micro', 'PMA_micro']])
            ]
    
    def extract_neonates_data(self,termness='term'):
        df = self.cohort_data
        self.diffusion_df = df[(df['termness'] == termness) & (
            ~df[['GA_diff',f'{self.cohort_name}_Anc_PC1' , 'Gender']].isna().any(axis=1))]
        self.volumetric_df = df[(df['termness'] == termness) & (
            ~df[['GA_vol', f'{self.cohort_name}_Anc_PC1' , 'Gender']].isna().any(axis=1))]
        self.micro_df = df[(df['termness'] == termness) & (~df[[
            'GA_micro', f'{self.cohort_name}_Anc_PC1' , 'Gender', 'l94_FA', 'l94_MD', 'l94_T12',
            'l94_T2', 'l94_FISO'
        ]].isna().any(axis=1))]
        
        
    @staticmethod
    def combine_imaging_genetic_data(imaging_df,genetic_df):
        return pd.concat([imaging_df,genetic_df],join='inner',axis=1)
    
    @staticmethod
    def preprocess_PRSice_PRS_Anc_files(file_path:str,
                                        ID_prefix: str=None, 
                                        ID_suffix: str= None,
                                        column_prefix: str = None,
                                        column_suffix: str = None) -> pd.core.frame.DataFrame:
        """[Preprocess PRS and ancestral PC tables with FID and IID columns]

        Args:
            file_path (str): [path to the txt file]
            ID_prefix (str): [additional info to add as prefix to ID]
            ID_suffix (str): [additional info to add as suffix to ID]
        Returns:
            pd.core.frame.DataFrame: [table with FID removed, ID sorted, matched with the babies ID]
        """
        
        table=pd.read_table(file_path,delim_whitespace=True)
        if ID_prefix is not None:
            table['IID']=[ID_prefix+str(i) for i in table['IID']]
        if ID_suffix is not None:
            table['IID']=[str(i)+ID_suffix for i in table['IID']]
        table=table.drop('FID',axis=1)
        table=table.rename({'IID':'ID'},axis=1)
        table=table.sort_values('ID').reset_index(drop=True)
        table['ID'] = [i.split('-')[0] for i in table['ID']]
        table = table.set_index('ID')
        if column_prefix is not None:
            table.columns = [column_prefix+i for i in table.columns]
        if column_suffix is not None:
            table.columns = [i+column_suffix for i in table.columns]
        return table
    
    @staticmethod
    def remove_outliers(df:pd.DataFrame,
                        ancestry_PCs=None,
                        by_name:Union[list,str]=None,plot=False):
        if not by_name:
            outliers = df.index[
                (np.abs(df[ancestry_PCs[0]].agg(stats.zscore)) >= 3.5) |
                (np.abs(df[ancestry_PCs[1]].agg(stats.zscore)) >= 3.5)]

            fig, ax = plt.subplots()
            for unique_cohort in df['cohort'].unique():
                ax.scatter(df.loc[df['cohort'] == unique_cohort, ancestry_PCs[0]],
                        df.loc[df['cohort'] == unique_cohort, ancestry_PCs[1]],label=unique_cohort)
                
            for i, txt in enumerate(outliers):
                ax.annotate(txt, (df.loc[outliers[i], ancestry_PCs[0]],
                                df.loc[outliers[i], ancestry_PCs[1]]))
            ax.legend()
            return df.drop(index=outliers), outliers
        else:
            return df.drop(index=by_name)

    
    


def preprocess_plink_assoc_linear_files(file_path:str)-> pd.core.frame.DataFrame:
    """[preprocess_plink_assoc_linear_files]

    Args:
        file_path ([str]): [path of the plink association file]

    Returns:
        pd.core.frame.DataFrame: [association file containinig single SNP in each row]
    """
    df = pd.read_csv(filepath=file_path,delim_whitespace=True)
    return df

def preprocess_magma_annotation_file(file_path:str) -> pd.core.frame.DataFrame:
    """[preprocess_magma_annotation_file, annotate gene to snps and snps to gene]

    Args:
        file_path (str): [the magma file location]

    Returns:
        pd.core.frame.DataFrame: [two tables, one gene to snps, and one snps to gene]
    """
    df = pd.read_csv(file_path,header=None,skiprows=[0,1]) # the first two rows of the magma file is window_size up and down value
    rows = [df.iloc[i,:].values[0].split('\t') for i in range(len(df))]
    
    annot_SNP_to_gene = defaultdict(dict)
    annot_gene_to_SNP = defaultdict(dict)
    
    for row in rows:
        gene_ID = int(row[0])
        annot_SNP_to_gene[gene_ID] = defaultdict(list)
        chromosome_start_stop = row[1].split(':')
        chromsome = int(chromosome_start_stop[0])
        start = int(chromosome_start_stop[1])
        stop = int(chromosome_start_stop [2])
        NSNPs = len(row[2:])
        annot_SNP_to_gene[gene_ID]['CHR'] = chromsome
        annot_SNP_to_gene[gene_ID]['START'] = start
        annot_SNP_to_gene[gene_ID]['STOP'] = stop
        annot_SNP_to_gene[gene_ID]['NSNP'] = NSNPs
        
        for snp in row[2:]:
            if snp not in annot_gene_to_SNP:
                annot_gene_to_SNP[snp] = defaultdict(list)
            annot_gene_to_SNP[snp]['Genes_list'].append(gene_ID)
            annot_gene_to_SNP[snp]['N_Genes'] = len(annot_gene_to_SNP[snp]['Genes_list'])
        
    annot_SNP_to_gene = pd.DataFrame(annot_SNP_to_gene).T
    annot_SNP_to_gene = annot_SNP_to_gene.reset_index()
    annot_SNP_to_gene.columns = ['ID','CHR','START','STOP','NSNP']
    
    annot_gene_to_SNP = pd.DataFrame(annot_gene_to_SNP).T
    annot_gene_to_SNP = annot_gene_to_SNP.reset_index()
    annot_gene_to_SNP.columns = ['SNP_ID','Genes_list','N_Genes']
    
    return annot_SNP_to_gene, annot_gene_to_SNP



def preprocess_david_gene_functional_classification_files(gene_func_class: pd.DataFrame)-> List[pd.DataFrame]:
    df = gene_func_class.copy()
    df.columns = ['Gene_ID','Gene_Name']
    indices = df.loc[[bool(re.search('Gene Group',i)) for i in df['Gene_ID']],:].index
    df['Gene_group']=None
    df['Enrichment_score']=None
    df['Gene_description'] = None
    for n_idx in range(len(indices)):
        if indices[n_idx+1] == indices[-1]:
            df.loc[indices[n_idx]:,'Gene_group'] = df.loc[indices[n_idx],'Gene_ID']
            df.loc[indices[n_idx]:,'Enrichment_score'] = float(df.loc[indices[n_idx],'Gene_Name'].split(' ')[2])
            df.loc[indices[n_idx]:,'Gene_description'] = df.loc[indices[n_idx]+2,'Gene_Name']
            break
        else:
            df.loc[indices[n_idx]:indices[n_idx+1],'Gene_group'] = df.loc[indices[n_idx],'Gene_ID']
            df.loc[indices[n_idx]:indices[n_idx+1],'Enrichment_score'] = float(df.loc[indices[n_idx],'Gene_Name'].split(' ')[2])
            df.loc[indices[n_idx]:indices[n_idx+1],'Gene_description'] = df.loc[indices[n_idx]+2,'Gene_Name']
    indices_entrez = df.loc[[bool(re.search('ENTREZ_GENE_ID',i)) for i in df['Gene_ID']],:].index
    to_remove_indices = list(indices_entrez) + list(indices)
    gene_description = list(df['Gene_description'].iloc[indices+2])
    summary_df = df.loc[indices].reset_index(drop=True)
    summary_df = summary_df.drop(['Gene_ID','Gene_Name'],axis=1)
    summary_df['Gene_description'] = [s[s.find("(")+1:s.find(")")] for s in summary_df['Gene_description']]

    df = df.drop(to_remove_indices).reset_index(drop=True)
    return df, summary_df

def generate_gmt_file(df:pd.DataFrame,
                      gene_nr_col:Union[str,int]=0,
                      gene_set_col:Union[str,int]=1,
                      out:str=None):
    
    if isinstance(gene_nr_col, str):
        gene_nr_col = [idx for idx,i in enumerate(df.columns) if i==gene_nr_col][0]
    if isinstance(gene_set_col, str):
        gene_set_col = [idx for idx,i in enumerate(df.columns) if i==gene_set_col][0]
    gene_sets = df.iloc[:,[gene_nr_col,gene_set_col]].copy()
    gene_sets.columns = ['gene_nr','gene_set']
    
    
    gene_sets['gene_nr'] = gene_sets['gene_nr'].astype('int64') # if it is float it will be int
    gene_sets['gene_nr'] = gene_sets['gene_nr'].astype('str') # we need it to be string to print
    gene_sets['gene_set'] = gene_sets['gene_set'].astype('str')
    
    gene_sets_dict = gene_sets.groupby('gene_set').agg(lambda x: x.tolist()).to_dict('index')
    if isinstance(out,str):
        with open(out,'a') as f:
            for gene_set in gene_sets_dict.keys():
                to_print = '\t'.join([gene_set] + [str(gene) for gene in gene_sets_dict[gene_set]['gene_nr']])
                f.writelines(to_print)
                f.writelines('\n')
    else:
        for gene_set in gene_sets_dict.keys():
            to_print = '\t'.join([gene_set] + [str(gene) for gene in gene_sets_dict[gene_set]['gene_nr']])
            print(to_print)
            
    
    
    
    
    
    
    
    
    
    
        
    
    
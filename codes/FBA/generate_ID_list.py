#!/usr/bin/env python

import argparse
from typing import List
import glob
import sys
from collections import defaultdict
import pandas as pd
import numpy as np

def main():

    parser = argparse.ArgumentParser(description='Get a list of subject IDs')
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser_file= parent_parser.add_argument_group('file')
    parent_parser_file.add_argument('--file',help='csv file name',type=str)
    parent_parser_file.add_argument('--header',help='header location',type=int)
    parent_parser_file.add_argument('--sep',help='csv delimiter',type=str,default=',')
    parent_parser_file.add_argument('--delim_whitespace',help='csv delim_whitespace',action='store_true')
    
    parent_parser_folder = parent_parser.add_argument_group('folder')
    parent_parser_folder.add_argument('--folder',help='Folder name',type=str)
    
    subparsers = parser.add_subparsers()
    id_generator = subparsers.add_parser('generate', parents=[parent_parser], help='generate IDs from folders or csv file')
    
    id_generator.add_argument('--idcolumns',help='csv columns index to select',nargs='+')
    id_generator.add_argument('--prefix',help='add prefix to columns name',type=str,nargs='+')
    id_generator.add_argument('--apcolumns',help='append other columns by index',nargs='+')

    id_generator.add_argument('--pattern',help='pattern to select',type=str,nargs='+',default='*')
    id_generator.add_argument('--duplicates',help='keep duplicated IDs',action='store_true')
    id_generator.add_argument('--no-duplicates',dest='duplicates',action='store_false')
    id_generator.add_argument('--get-differences',dest='differences',help='get differences between current and previous list',action='store_true')
    id_generator.add_argument('--out',help='Output txt',type=str)
    
    id_select = subparsers.add_parser('select', parents=[parent_parser],help='apply criteria to a generated ID list')
    id_select.add_argument('--criteria', help='columns followed by criteria to select from', type=str,nargs='+')
    id_select.add_argument('--group',help='denote the selected id group',type=str)
    id_select.add_argument('--remove',help='remove entries by ID',type=str,nargs='+')
    id_select.add_argument('--out',help='Output txt',type=str)
    
    id_matrix = subparsers.add_parser('matrix',parents=[parent_parser],help='generate design and contrast matrix for fba')
    id_matrix.add_argument('--categorical',help='denote columns containing categorical variable',nargs='+',type=int)
    id_matrix.add_argument('--continuous',help='denote columns containing continuous variables',nargs='+',type=int)
    id_matrix.add_argument('--standardize', help='standardize the continuous variables',action='store_true')
    id_matrix.add_argument('--no-standardize',dest='standardize',action='store_false')
    id_matrix.add_argument('--intercept',help='add columns of intercepts',action='store_true')
    id_matrix.add_argument('--no-intercept',dest='intercept',action='store_false')
    id_matrix.add_argument('--contrast',help='define the column to contrast',nargs='+')
    id_matrix.add_argument('--neg', help='whether to perform opposite hypothesis. Default False', action='store_true')
    id_matrix.add_argument('--both', help='whether to perform both positive and negative hypothesis. Default False', action='store_true')
    id_matrix.add_argument('--catnames',help='assign column names to categorical variable',nargs='+',type=str)
    id_matrix.add_argument('--contnames',help='assign column names to continuous variables',nargs='+',type=str)
    id_matrix.add_argument('--id_prefix',help='assign prefix to the IDs in the ID file',type=str)
    id_matrix.add_argument('--id_suffix',help='assign suffix to the IDs in the ID file',type=str)
    id_matrix.add_argument('--generate_vest',help='Vest file is needed for randomise', action='store_true')
    id_matrix.add_argument('--sort_id',help='sort the ids alphabeto-numerically',action='store_true')
    id_matrix.add_argument('--out_ID',help='output for ID file')
    id_matrix.add_argument('--test',help='whether to just print to the terminal',action='store_true')

    args=parser.parse_args()

    return args




class Generateids:

    @staticmethod
    def generate_IDs(filename:str=None,
                     foldername:str=None,
                     sep:str=None,
                     delim_whitespace:bool=False,
                     header:int=None,
                     pattern:List[str]=None,
                     idcolumns:List[int]=None,
                     apcolumns:List[int]=None,
                     prefix:List[str]=None,
                     previous_IDs:List[str] = None,
                     duplicated:bool=False,
                     get_differences:bool=False,
                     out:str=None):
        """
        Parameters
        ----------
        filename : str, optional
            csv file, where the first column is the subject of interest. The default is None.
        foldername : str, optional
            directory, where the folder name is the subject of interest. The default is None.
        sep : str, optional
            delimiter of the csv file. The default is ','.
        pattern : List[str], optional
            pattern to select the IDs of interest. The default is None.
        columns : List[int], optional
            select the id column names, e.g. ID and SES columns. The default is None.
        apcolumns : List[int], optional
            select categorical and continuous columns to append to the ID column. The default is None.
        prefix : List[str], optional
            prefix to add to each of the id columns, e.g. sub- and ses-. The default is None.
        previous_IDs : List[str], optional
            Use with pipe in shell, if previous output is passed then compare current list with that list. The default is None.
        duplicated: bool, optional
            If False, remove duplicated IDs in the first column, else keep. The default is False
        out : str, optional
            output file. The default is None.

        Returns
        -------
        None.

        """
        if isinstance(filename,str):
            if delim_whitespace:
                file = pd.read_csv(filename,header=header,comment='#',delim_whitespace=True)
            elif isinstance(sep,str):
                file = pd.read_csv(filename,header=header,comment='#',sep=sep)
            ID_pd=pd.DataFrame()
            
            if isinstance(idcolumns,list):
                if prefix is None:
                    prefix = ['' for i in range(len(idcolumns))]
                else:
                    if len(prefix) == 1:
                        prefix = [prefix for i in range(len(idcolumns))]
                if not duplicated:
                    file = file.drop_duplicates(subset=file.columns[0],keep='last').reset_index(drop=True)
                if idcolumns[0].isdigit():
                    idcolumns = [int(i) for i in idcolumns]
                    for idx,(col,pref) in enumerate(zip(idcolumns,prefix)):
                        if isinstance(file.iloc[0,col],float):
                            file.iloc[:,col] = file.iloc[:,col].astype('int')
                        ID_pd[idx] = pref + file.iloc[:,col].astype('str')
                else:
                    for idx,(col,pref) in enumerate(zip(idcolumns,prefix)):
                        if isinstance(file.loc[0,col],float):
                            file.loc[:,col] = file.loc[:,col].astype('int')
                        ID_pd[idx] = pref + file.loc[:,col].astype('str')
                ID_pd['ID'] = ID_pd[ID_pd.columns].agg('/'.join,axis=1)
                ID_pd = ID_pd.drop(columns=[i for i in ID_pd.columns if i != 'ID'])

            else:
                idcolumns=[]
                #if no columns provided, default the first column is the ID
                ID_pd = file.copy()
                ID_pd.columns = ['ID'] + ID_pd.columns.tolist()[1:]

            if isinstance(apcolumns,list):
                if apcolumns[0].isdigit():
                    apcolumns = [int(i) for i in apcolumns]
                    for col in apcolumns:
                        ID_pd = pd.concat([ID_pd,file.loc[:,file.columns[col]]],axis=1)
                else:
                    ID_pd = pd.concat([ID_pd,file.loc[:,apcolumns]],axis=1)
            else:
                ID_pd = ID_pd[['ID']].copy()
            ID_pd = ID_pd.astype(str)
            ID_list = [','.join(ID_pd.iloc[row,:].tolist()) for row in range(len(ID_pd))]

        if isinstance(foldername,str):
            print(foldername)
            if foldername[-1] == '/':
                foldername = foldername[:-1]
            if isinstance(pattern,list):
                new_pattern ='/'.join(pattern) # a/b/c
            pattern_to_search ='/'.join([foldername,new_pattern])
            print(pattern_to_search)
            pattern_list = glob.glob(pattern_to_search)
            ID_list = [i.replace(foldername+'/','') for i in pattern_list]
            if not duplicated:
                folder=defaultdict(list)
                try:
                    for i,j in map(lambda x: x.split('/'),ID_list):
                        if len(folder[i])>1:
                            folder[i]=[]
                        folder[i].append(j)
                    ID_list = ['/'.join([i,j[0]]) for i,j in folder.items()]
                except ValueError:
                    pass

        if isinstance(previous_IDs,list):
            new_list = Generateids.get_common_IDs(ID_list,previous_IDs,get_differences=get_differences)
            ID_list = [','.join(new_list.iloc[row,:].tolist()) for row in range(len(new_list))]
        
        if out is not None:
            with open(out,'a') as f:
                for i in ID_list:
                    f.writelines(i)
                    f.writelines('\n')
        else:
            for i in ID_list:
                print(i)

    @staticmethod
    def get_common_IDs(list1:List[str],list2:List[str],list1_delimited:str=',',list2_delimited:str=',',get_differences=False):
        """
        Use when previous IDs is passed.

        Parameters
        ----------
        list1 : List[str]
            current ID list.
        list2 : List[str]
            previous ID list.
        list1_delimited : str, optional
            delimiter of list1. The default is ','.
        list2_delimited : str, optional
            delimiter of list2. The default is ','.
        get_difference: bool
            select only values unique to list1, but not list 2. The default is False
            
        Returns
        -------
        common_list : pd.DataFrame
            new common dataframe between the two.

        """
        # you can only get the common lines in the first columns
        list1_IDs = pd.DataFrame([ID.split(list1_delimited) for ID in list1])
        list2_IDs = pd.DataFrame([ID.split(list2_delimited) for ID in list2])
        
        list1_IDs.columns=['IDs']+list1_IDs.columns.tolist()[1:]
        list2_IDs.columns=['IDs']+list2_IDs.columns.tolist()[1:]

        if get_differences:
            result_list = list1_IDs[~list1_IDs.IDs.isin(list2_IDs.IDs)].reset_index(drop=True)
        else:
            result_list= pd.merge(list1_IDs,list2_IDs,on ='IDs',how = 'inner')
        
        return result_list

    @staticmethod
    def select_IDs(filename:str=None,
                   sep:str=None,
                   delim_whitespace:bool=False,
                   header:int=None,
                   criteria:List[str]=None,
                   group:str=None,
                   previous_IDs:str=None,
                   remove_IDs:List[str]=None,
                   out:str=None
                   ):
        """

        Parameters
        ----------
        filename : str, optional
            csv file name. The default is None.
        sep : TYPE, optional
            delimiter of the csv file. The default is ','.
        criteria : List[str], optional
            takes 3 values such that the first value is the column index,
            second value is the number to compare the column values to
            third value is the operator. The default is None.
        group : str, optional
            assign the rows that pass the criteria to a name, e.g. term. The default is None.
        previous_IDs : str, optional
            if previous ID is passed then append the current one with the previous one. The default is None.
        remove_IDs: str, optional
            remove subset of individuals by their IDs. The default is None.
        out : str, optional
            output destination. The default is None.

        Returns
        -------
        None.

        """
        if isinstance(filename,str):
            if delim_whitespace:
                file = pd.read_csv(filename,header=header,comment='#',delim_whitespace=True)
            elif isinstance(sep,str):
                file = pd.read_csv(filename,header=header,comment='#',sep=sep)
        updated_ID_pd = file.copy()

        if isinstance(criteria,list):
            commands =(criteria[i:i+3] for i in range(0,len(criteria),3))
            for column,value,operation in commands:
                if not value.isdigit():
                    try:
                        float(value)
                    except ValueError:
                        value ="'{}'".format(value)
                command = 'updated_ID_pd.iloc[:,'+column+']'+operation+value
                print(command)
                evaluation = eval(command)
                updated_ID_pd = updated_ID_pd[evaluation]
        updated_ID_pd = updated_ID_pd.astype(str)
        if isinstance(group,str):
            updated_ID_pd['group'] = group
        ID_list = [','.join(updated_ID_pd.iloc[row,:].tolist()) for row in range(len(updated_ID_pd))]
        if isinstance(previous_IDs,list):
            ID_list+=previous_IDs
        
        if isinstance(remove_IDs,list):
            for remove_subj in remove_IDs:
                ID_list = [ID_line for ID_line in ID_list if remove_subj not in ID_line]
            
        if out is not None:
            with open(out,'a') as f:
                for i in ID_list:
                    f.writelines(i)
                    f.writelines('\n')
        else:
            for i in ID_list:
                print(i)
    
    @staticmethod
    def get_design_contrast_matrix(filename:str=None,
                                   sep:str=None,
                                   delim_whitespace:bool=False,
                                   header:int=None,
                                   categoricalVariable:List[int]=None,
                                   continuousVariable:List[int]=None,
                                   standardize:bool=True,
                                   intercept:bool=True,
                                   contrast:int=None,
                                   f_stats:bool=False,
                                   categorical_Names:List[str]=None,
                                   continuous_Names:List[str]=None,
                                   negative:bool=False,
                                   both:bool=False,
                                   id_prefix:str=None,
                                   id_suffix:str=None,
                                   sort_id:bool=False,
                                   generate_vest:bool=False,
                                   ID_file:str=None,
                                   f_file:str=None,
                                   test:bool=False):
        """
        Parameters
        ----------
        filename : str, optional
            csv file, where the first input is the sub-CCXXX/ses-XXX. The default is None.
        categoricalVariable : TYPE, optional
            the column index of the categorical variables in the filename. The default is None.
        continuousVariable : TYPE, optional
            the column index of the continuous variables in the filename. The default is None.
        standardize : bool, optional
            Bool - to standardize the continuous variables using following method:
            ( observed - mean )/ (stadandard deviation). The default is True.
        contrast : int, optional
            the column index of the variable in the filename we are contrasting for. The default is None.
        categorical_Names : List[str], optional
            assign column name of the categorical variables. The default is None.
        continuous_Names : TYPE, optional
            assign column name of the continuous variables. The default is None.
        id_prefix: str, optional
            define prefix to the id in the id list. (useful when need to define directory)
        id_suffix: str, optional
            define suffix to the id in the id list.
        ID_file : TYPE, optional
            output location of the ID file, where it is a file containing the 
            name of the subjects used in the study in the format sub-CCXXX_ses-XXX.mif. 
            The default is None.
        contrast_file : str, optional
            output location of the contrast matrix,e.g. 0 1 0 0 0. The default is None.
        design_file : TYPE, optional
            output location of the design matrix, where each line contains the 
            values of indepdendent variables. The default is None.

        Returns
        -------
        None.

        """
        if isinstance(filename,str):
            if delim_whitespace:
                ID_pd = pd.read_csv(filename,header=header,comment='#',delim_whitespace=True)
            elif isinstance(sep,str):
                ID_pd = pd.read_csv(filename,header=header,comment='#',sep=sep)
        if id_prefix is None:
            id_prefix=''
        if id_suffix is None:
            id_suffix=''
        if sort_id:
            ID_pd=ID_pd.sort_values(ID_pd.columns[0]).reset_index(drop=True)
        name_file = ID_pd.iloc[:,0].apply(lambda x:id_prefix+x.replace('/','_')+id_suffix+'.mif')
        ID_list = name_file.tolist()
        if isinstance(ID_file,str):
            if '/' in ID_file:
                prefix='/'.join(ID_file.split('/')[0:-1]) + '/'
            else:
                prefix=''
            with open(ID_file,'w') as f:
                for i in ID_list:
                    f.writelines(i)
                    f.writelines('\n')
        ID_pd.columns = [str(i) for i in ID_pd.columns.tolist()]
        design_pd = pd.DataFrame()
        contrast_matrix=[]
        if isinstance(categoricalVariable,list):
            category_pd = pd.DataFrame()
            if not isinstance(categorical_Names,list):
                categorical_Names = [f'Categorical_{i}' for i in categoricalVariable]
            for category,name in zip(categoricalVariable,categorical_Names):
                temp_category = pd.get_dummies(ID_pd.iloc[:,category],dtype=int)
                temp_category.columns = [f'{name}_{i}' for i in temp_category.columns.tolist()]
                #selecting one variable as the reference data
                #in the other columns, where the rows in which the reference 
                # data appears, is set to -1. see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/GLM#ANOVA:_1-factor_4-levels__.281-way_between-subjects_ANOVA.29 
                #here the first column is selected as the reference.
                reference_column = temp_category.columns[0]
                for col in temp_category.columns[1:]:
                    temp_category[col] = [-1 if temp_category.loc[idx,reference_column] == 1 else temp_category.loc[idx,col] for idx in temp_category.index]
                category_pd = pd.concat([category_pd,temp_category.iloc[:,1:]],axis=1)
        else:
            categoricalVariable = []
            category_pd = pd.DataFrame()

        if isinstance(continuousVariable,list):
            if not isinstance(continuous_Names,list):
                continuous_Names = [f'Continuous_{i}' for i in continuousVariable]
            continuous_pd = ID_pd.iloc[:,continuousVariable]
            continuous_pd.columns = continuous_Names
            if standardize:
                continuous_pd = continuous_pd.apply(lambda x: [(i - np.mean(x))/np.std(x) for i in x])
        else:
            continuousVariable = []
            continuous_pd = pd.DataFrame()
        
        if intercept:
            design_pd['intercept'] = [1 for i in range(len(ID_pd))]
        design_pd = pd.concat([design_pd,category_pd,continuous_pd],axis=1)
        independentVariable=['intercept'] + categoricalVariable+continuousVariable if intercept else categoricalVariable + continuousVariable
        independentVariable_names = design_pd.columns.tolist()
        # these two variables can have different length, if there are more than 1 group in the categorical data
        if not isinstance(contrast,list):
            raise TypeError('Contrast is empty')
        for hypothesis in contrast:
            contrast_to_not_consider = [i for i in contrast if i != hypothesis]
            if isinstance(ID_file,str) or test:
                temp_design_pd = design_pd.drop(columns=[i for n in contrast_to_not_consider 
                                                        for i in independentVariable_names
                                                        if n in i]).astype('str')
                design_matrix = [' '.join(temp_design_pd.iloc[row,:].tolist()) for row in range(len(design_pd))]
                if not test:
                    with open(prefix+hypothesis.replace('.','_')+'_design.txt','w') as f:
                        if generate_vest:
                            f.writelines(f'/NumWaves {len(independentVariable)-len(contrast)+1}\n')
                            f.writelines(f'/NumPoints {len(temp_design_pd)}\n')
                            f.writelines('\n')
                            f.writelines('/Matrix')
                        else:
                            f.writelines(['#']+[str(i)+' ' for i in temp_design_pd.columns])
                        f.writelines('\n')
                        for line in design_matrix:
                            f.writelines(line)
                            f.writelines('\n')
                else:
                    print(prefix+hypothesis.replace('.','_')+'_design.txt')
                    print(temp_design_pd.head(2))
                        
                contrast_matrix_temp = [0 for i in range(len(temp_design_pd.columns))]
                contrast_id = [idx for idx,i in enumerate(temp_design_pd.columns.tolist()) if i == hypothesis]
                if len(contrast_id) == 0: # if nothing is in it
                    #this happens if it is categorical var
                    contrast_id = [idx for idx,i in enumerate(temp_design_pd.columns.tolist()) if hypothesis in i]
                if len(contrast_id) > 1:
                    raise ValueError('the term of interest is repeating more than once')
                try:
                    if negative: # test opposite hypothesis
                        contrast_matrix_temp[contrast_id[0]] = -1
                    elif both:
                        contrast_matrix_temp[contrast_id[0]] = -1
                        contrast_matrix_temp2 = [i for i in contrast_matrix_temp]
                        contrast_matrix_temp2[contrast_id[0]] = 1
                    else:
                        contrast_matrix_temp[contrast_id[0]] = 1
                except IndexError:
                    raise IndexError("Check again that maybe one of the contrast name is a substring of another contrast name or the contnames do not match contrast name")    
                if not test:
                    with open(prefix+hypothesis.replace('.','_')+'_contrast.txt','w') as f:
                        if generate_vest:
                            f.writelines(f'/NumWaves {len(independentVariable)}\n')
                            f.writelines(f'/NumContrasts {len(contrast_matrix)}\n')
                            f.writelines('\n')
                            f.writelines('/Matrix')
                            f.writelines('\n')
                        f.writelines(' '.join([str(i) for i in contrast_matrix_temp]))
                        f.writelines('\n')
                        if both:
                            f.writelines(' '.join([str(i) for i in contrast_matrix_temp2]))
                            f.writelines('\n')
                else:
                    print(hypothesis.replace('.','_'))
                    print(contrast_matrix_temp)
                    if both:
                        print(contrast_matrix_temp2)

if __name__ == '__main__':
    args = main()
    #if output is passed can be used with piping in bash
    if not sys.stdin.isatty():
        #output is passed
        tmp_previous_IDs = sys.stdin.readlines()
        previous_IDs = [i.replace('\n','') for i in tmp_previous_IDs]
    else:
        previous_IDs = None
    if sys.argv[1] == 'generate':
        Generateids.generate_IDs(filename=args.file,
                             foldername=args.folder,
                             header=args.header,
                             sep=args.sep,
                             delim_whitespace=args.delim_whitespace,
                             pattern=args.pattern,
                             idcolumns=args.idcolumns,
                             prefix=args.prefix,
                             apcolumns=args.apcolumns,
                             previous_IDs=previous_IDs,
                             duplicated=args.duplicates,
                             get_differences=args.differences,
                             out=args.out)

    elif sys.argv[1] == 'select':
        Generateids.select_IDs(filename=args.file,
                               sep=args.sep,
                               delim_whitespace=args.delim_whitespace,
                               header=args.header,
                               criteria=args.criteria,
                               group=args.group,
                               previous_IDs=previous_IDs,
                               remove_IDs=args.remove,
                               out=args.out)

    elif sys.argv[1] == 'matrix':
        Generateids.get_design_contrast_matrix(filename=args.file,
                                               sep=args.sep,
                                               delim_whitespace=args.delim_whitespace,
                                               header=args.header,
                                               categoricalVariable=args.categorical,
                                               continuousVariable=args.continuous,
                                               standardize=args.standardize,
                                               intercept=args.intercept,
                                               contrast=args.contrast,
                                               categorical_Names=args.catnames,
                                               continuous_Names=args.contnames,
                                               test=args.test,
                                               negative=args.neg,
                                               both=args.both,
                                               id_prefix=args.id_prefix,
                                               id_suffix=args.id_suffix,
                                               sort_id=args.sort_id,
                                               ID_file=args.out_ID,
                                               generate_vest=args.generate_vest)
    


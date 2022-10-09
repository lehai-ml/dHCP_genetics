#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from typing import List
import glob
import sys


def main():

    parser = argparse.ArgumentParser(description='Get a list of subject IDs')
    subparsers = parser.add_subparsers()
    id_generator = subparsers.add_parser('generate', help='generate IDs from folders or csv file')
    file_group = id_generator.add_argument_group('file')
    folder_group = id_generator.add_argument_group('folder')
    file_group.add_argument('--file',help='csv file name',type=str)
    file_group.add_argument('--sep',help='csv delimiter',type=str,nargs='?',default=',')
    file_group.add_argument('--columns',help='csv columns index to select',type=int,nargs='+')
    file_group.add_argument('--prefix',help='add prefix to columns name',type=str,nargs='+')
    file_group.add_argument('--apcolumns',help='append other columns by index',type=int,nargs='+')

    folder_group.add_argument('--folder',help='Folder name',type=str)
    folder_group.add_argument('--pattern',help='pattern to select',type=str,nargs='+',default='*')

    id_generator.add_argument('--out',help='Output txt',type=str)

    id_select = subparsers.add_parser('select', help='apply criteria to a generated ID list')
    id_select.add_argument('--file',help='csv file name', type=str)
    id_select.add_argument('--sep',help='csv delimiter',type=str,nargs='?',default=',')
    id_select.add_argument('--criteria', help='columns followed by criteria to select from', type=str,nargs='+')
    id_select.add_argument('--group',help='denote the selected id group',type=str)
    id_select.add_argument('--out',help='Output txt',type=str)
    
    id_matrix = subparsers.add_parser('matrix', help='generate design and contrast matrix for fba')
    id_matrix.add_argument('--file',help='csv file name', type=str)
    id_matrix.add_argument('--categorical',help='denote columns containing categorical variable',nargs='+',type=int)
    id_matrix.add_argument('--continuous',help='denote columns containing continuous variables',nargs='+',type=int)
    id_matrix.add_argument('--standardize', help='standardize the continuous variables',action='store_true')
    id_matrix.add_argument('--no-standardize',dest='standardize',action='store_false')
    id_matrix.add_argument('--contrast',help='define the column to contrast',type=int)
    id_matrix.add_argument('--catnames',help='assign column names to categorical variable',nargs='+',type=str)
    id_matrix.add_argument('--contnames',help='assign column names to continuous variables',nargs='+',type=str)
    id_matrix.add_argument('--out_ID',help='output for ID file')
    id_matrix.add_argument('--out_design',help='output for design matrix')
    id_matrix.add_argument('--out_contrast',help='output for contrast matrix')

    args=parser.parse_args()

    return args




class Generateids:

    @staticmethod
    def generate_IDs(filename:str=None,
                     foldername:str=None,
                     sep:str=',',
                     pattern:List[str]=None,
                     columns:List[int]=None,
                     apcolumns:List[int]=None,
                     prefix:List[str]=None,
                     previous_IDs:List[str] = None,
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
        out : str, optional
            output file. The default is None.

        Returns
        -------
        None.

        """

        if isinstance(filename,str):
            file = pd.read_csv(filename,sep=',',header=None)
            ID_pd=pd.DataFrame()

            if columns is not None:
                if prefix is None:
                    prefix = ['' for i in range(len(columns))]
                else:
                    if len(prefix) == 1:
                        prefix = [prefix for i in range(len(columns))]
                for idx,(col,pref) in enumerate(zip(columns,prefix)):
                    ID_pd[idx] = pref + file.iloc[:,col].astype('str')

            ID_pd['ID'] = ID_pd[ID_pd.columns].agg('/'.join,axis=1)
            
            if apcolumns is not None:
                for col in apcolumns:
                    ID_pd[file.columns[col]] = file.loc[:,file.columns[col]].astype('str')
                ID_pd = ID_pd[['ID']+file.columns[apcolumns].tolist()].copy()
            else:
                ID_pd = ID_pd[['ID']].copy()
            ID_list = [','.join(ID_pd.iloc[row,:].tolist()) for row in range(len(ID_pd))]

        if isinstance(foldername,str):
            if isinstance(pattern,list):
                pattern ='/'.join(pattern) # a/b/c
            pattern_to_search ='/'.join([foldername,pattern])
            pattern_list = glob.glob(pattern_to_search)
            ID_list = [i.replace(foldername+'/','') for i in pattern_list]

        if isinstance(previous_IDs,list):
            common_list = Generateids.get_common_IDs(ID_list,previous_IDs) 
            ID_list = [','.join(common_list.iloc[row,:].tolist()) for row in range(len(common_list))]

        if out is not None:
            with open(out,'a') as f:
                for i in ID_list:
                    f.writelines(i)
                    f.writelines('\n')
        else:
            for i in ID_list:
                print(i)

    @staticmethod
    def get_common_IDs(list1:List[str],list2:List[str],list1_delimited:str=',',list2_delimited:str=','):
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
        
        common_list = pd.merge(list1_IDs,list2_IDs,on ='IDs',how = 'inner')
        return common_list

    @staticmethod
    def select_IDs(filename:str=None,
                   sep=',',
                   criteria:List[str]=None,
                   group:str=None,
                   previous_IDs:str=None,
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
        out : str, optional
            output destination. The default is None.

        Returns
        -------
        None.

        """
        if isinstance(filename,str):
            ID_pd = pd.read_csv(filename,sep=sep,header=None)
        else:
            ID_pd = pd.DataFrame([ID.split(sep) for ID in filename])
            #['4' '37' '-gt' 4 '37' '-lt']
        updated_ID_pd = ID_pd.copy()

        if isinstance(criteria,list):
            commands =(criteria[i:i+3] for i in range(0,len(criteria),3))
            for column,value,operation in commands:
                if not value.isdigit():
                    try:
                        float(value)
                    except ValueError:
                        value ="'{}'".format(value)
                command = 'updated_ID_pd.iloc[:,'+column+']'+operation+value
                evaluation = eval(command)
                updated_ID_pd = updated_ID_pd[evaluation]
        updated_ID_pd = updated_ID_pd.astype(str)
        if isinstance(group,str):
            updated_ID_pd['group'] = group
        ID_list = [','.join(updated_ID_pd.iloc[row,:].tolist()) for row in range(len(updated_ID_pd))]
        if isinstance(previous_IDs,list):
            ID_list+=previous_IDs

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
                            categoricalVariable:List[int]=None,
                            continuousVariable:List[int]=None,
                            standardize:bool=True,
                            contrast:int=None,
                            categorical_Names:List[str]=None,
                            continuous_Names:List[str]=None,
                            ID_file:str=None,
                            contrast_file:str=None,
                            design_file:str=None):
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
            ID_pd = pd.read_csv(filename,header=None)

        name_file = ID_pd[0].apply(lambda x:x.replace('/','_')+'.mif')
        ID_list = name_file.tolist()
        if isinstance(ID_file,str):
            with open(ID_file,'a') as f:
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
                temp_category[temp_category==0] = -1
                column_name = f'{temp_category.columns[0]}=1'
                temp_category.columns = [column_name] + [temp_category.columns[1]]
                category_pd = pd.concat([category_pd,temp_category.iloc[:,0]],axis=1)
                
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
        design_pd['intercept'] = [1 for i in range(len(ID_pd))]
        design_pd = pd.concat([design_pd,category_pd,continuous_pd],axis=1)
        independentVariable=categoricalVariable+continuousVariable
        
        contrast_matrix = [0 for i in range(len(independentVariable) +1 )]
        if isinstance(contrast,int):
            if contrast in independentVariable:
                contrast_id = [idx for idx,i in enumerate(independentVariable) if i == contrast]
                contrast_matrix[contrast_id[0]+1] = 1
        
        if isinstance(contrast_file,str):
            with open(contrast_file,'w') as f:
                for number in contrast_matrix:
                    f.write(str(number)+' ')
        if isinstance(design_file,str):
            design_pd = design_pd.astype('str')
            design_matrix = [' '.join(design_pd.iloc[row,:].tolist()) for row in range(len(design_pd))]
            with open(design_file,'w') as f:
                f.writelines(['#']+[str(i)+' ' for i in design_pd.columns])
                f.writelines('\n')
                for line in design_matrix:
                    f.writelines(line)
                    f.writelines('\n')



if __name__ == '__main__':
    args = main()
    #if output is passed can be used with piping in bash
    if not sys.stdin.isatty():
        #output is passed
        tmp_previous_IDs = sys.stdin.readlines()
        previous_IDs = [i.replace('\n','') for i in tmp_previous_IDs]
    else:
        previous_IDs = None
    if sys.argv[1] == 'select':
        Generateids.select_IDs(filename=args.file,
                               sep=args.sep,
                               criteria=args.criteria,
                               group=args.group,
                               previous_IDs=previous_IDs,
                               out=args.out)
    elif sys.argv[1] == 'generate':
        Generateids.generate_IDs(filename=args.file,
                             foldername=args.folder,
                             sep=args.sep,
                             pattern=args.pattern,
                             columns=args.columns,
                             prefix=args.prefix,
                             apcolumns=args.apcolumns,
                             previous_IDs=previous_IDs,
                             out=args.out)
    elif sys.argv[1] == 'matrix':
        Generateids.get_design_contrast_matrix(filename=args.file,
                                               categoricalVariable=args.categorical,
                                               continuousVariable=args.continuous,
                                               standardize=args.standardize,
                                               contrast=args.contrast,
                                               categorical_Names=args.catnames,
                                               continuous_Names=args.contnames,
                                               ID_file=args.out_ID,
                                               contrast_file=args.out_contrast,
                                               design_file=args.out_design)
    


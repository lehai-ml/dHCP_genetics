#!/usr/bin/env python
import argparse
import pandas as pd
from typing import List
import glob
import sys

parser = argparse.ArgumentParser(description='Get a list of subject IDs')
group1=parser.add_argument_group('file')
group2=parser.add_argument_group('folder')
group1.add_argument('--file',help='Csv file name',type=str)
group1.add_argument('--sep','-s',help='csv delimiter',type=str,nargs='?',default=',')
group1.add_argument('--columns','-co',help='csv columns index to select',type=int,nargs='+')
group1.add_argument('--prefix','-pr',help='add prefix to columns name',type=str,nargs='+')
group1.add_argument('--apcolumns','-aco',help='append other columns by index',type=int,nargs='+')

group2.add_argument('--folder',help='Folder name',type=str)
group2.add_argument('--pattern','-p',help='pattern to select',type=str,nargs='+',default='*')

parser.add_argument('--out',help='Output txt',type=str)


args=parser.parse_args()

#if output is passed can be used with piping in bash
if not sys.stdin.isatty():
    #output is passed
    tmp_previous_IDs = sys.stdin.readlines()
    previous_IDs = [i.replace('\n','') for i in tmp_previous_IDs]
else:
    previous_IDs = None

def get_common_IDs(list1:List[str],list2:List[str],list1_delimited=',',list2_delimited=','):
    # you can only get the common lines in the first columns
    list1_IDs = pd.DataFrame([ID.split(list1_delimited) for ID in list1])
    list2_IDs = pd.DataFrame([ID.split(list2_delimited) for ID in list2])

    list1_IDs.columns=['IDs']+list1_IDs.columns.tolist()[1:]
    list2_IDs.columns=['IDs']+list2_IDs.columns.tolist()[1:]
    
    common_list = pd.merge(list1_IDs,list2_IDs,on ='IDs',how = 'inner')
    return common_list

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
    Generate IDs 
    """

    if isinstance(filename,str):
        file = pd.read_csv(filename,sep=',')
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
        common_list = get_common_IDs(ID_list,previous_IDs) 
        ID_list = [','.join(common_list.iloc[row,:].tolist()) for row in range(len(common_list))]

    if out is not None:
        with open(out,'a') as f:
            for i in ID_list:
                f.writelines(i)
                f.writelines('\n')
    else:
        for i in ID_list:
            print(i)

generate_IDs(filename=args.file,
             foldername=args.folder,
             sep=args.sep,
             pattern=args.pattern,
             columns=args.columns,
             prefix=args.prefix,
             apcolumns=args.apcolumns,
             previous_IDs=previous_IDs,
             out=args.out)





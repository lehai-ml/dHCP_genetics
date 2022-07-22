#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used to analyse high dimension data
@author: lh20
"""

#Scikit-lib
#Python essentials

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components #use depth-first search
from typing import List, Union, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureReduction:
    
    
    def retain_non_zero_features(df:pd.DataFrame,
                                 dependentVar_cols:Optional[Union[List[str],
                                                 List[np.ndarray]]] = None,
                                 threshold:float=0.5)->pd.DataFrame:
        
        zero_perc_calc=lambda feature: np.sum(feature==0)/len(feature)
        dependentVars = df.loc[:,dependentVar_cols].copy()
        cols_to_drop = dependentVars.columns[dependentVars.apply(zero_perc_calc)>threshold]
        return df.drop(columns=cols_to_drop)
    
    
    @staticmethod
    def perform_PCA(df:Union[pd.DataFrame,np.ndarray],
                    dependentVar_cols:Optional[List[str]] = None,
                    n_components:int=None,
                    random_state:int=42,
                    scaling:bool=False,
                    columns = None):
        pca = PCA(n_components=n_components,
                  random_state=random_state)
        if dependentVar_cols is not None:
            if not isinstance(dependentVar_cols,list):
                dependentVar_cols = [dependentVar_cols]
        if isinstance(df,pd.DataFrame):
            if dependentVar_cols is None:
                X = df.to_numpy()
            else:
                X = df[dependentVar_cols].to_numpy()
        elif isinstance(df,np.ndarray):
            X = df.copy()
        if scaling:
            X = StandardScaler().fit_transform(X)
        X_pca = pca.fit_transform(X)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        if dependentVar_cols is not None:
            loading_matrix = pd.DataFrame(loadings, index = dependentVar_cols)
        else:
            loading_matrix = pd.DataFrame(loadings)
        return pca, X_pca, loading_matrix    
    
    def combine_columns_together(df:pd.DataFrame,
                                 group_columns:Union[dict,List],
                                 operation:str = 'sum',
                                 remove_duplicated:bool=True):
        """
        Combine the columns by performing an operation on it.

        Parameters
        ----------
        df : pd.DataFrame
            Data Frame of interest.
        group_columns_dict : dict
            Dictionary {'new grouped name': [ list of columns names need to be grouped]}.
            new grouped name cannot be the same name as the original names
        operation : str, optional
            {'sum','mean'}. The default is 'sum
        remove_duplicated : bool, optional
            If you don't want to use the new grouped name, but only update the original columns. The default is True.
            if passing list, then drop the columns
            Useful when you want to plot

        Returns
        -------
        temp_df : pd.DataFrame
            New updated df

        """
        temp_df = df.copy()
        if isinstance(group_columns, dict):
            for group,column_names in group_columns.items():
                if operation == 'sum':
                    temp_df[group] = temp_df[column_names].sum(axis=1)
                elif operation == 'mean':
                    temp_df[group] = temp_df[column_names].mean(axis=1)
                if remove_duplicated:
                    temp_df.drop(columns = column_names,inplace=True)
                else:
                    for column in column_names:
                        temp_df[column] = temp_df[group]
                    temp_df.drop(columns = group, inplace=True)
        elif isinstance(group_columns,list):
            if not isinstance(group_columns[0],list):
                raise AttributeError('pass list of list if have only 1 group')
            for column_names in group_columns:
                if operation == 'sum':
                    temp_df['temp'] = temp_df[column_names].sum(axis=1)
                elif operation == 'mean':
                    temp_df['temp'] = temp_df[column_names].mean(axis=1)
                for column in column_names:
                    temp_df[column] = temp_df['temp']
                temp_df.drop(columns = 'temp', inplace=True)
                if remove_duplicated:
                    temp_df.drop(columns=column_names[1:],inplace=True)
        return temp_df

        
    
    
    
    
    
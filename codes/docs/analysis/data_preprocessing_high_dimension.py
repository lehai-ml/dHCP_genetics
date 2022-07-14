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
    def perform_PCA(df:pd.DataFrame,
                    dependentVar_cols:Optional[Union[List[str],
                                    List[np.ndarray]]] = None,
                    n_components:int=None,
                    random_state:int=42,
                    scaling:bool=False,
                    columns = None):
        pca = PCA(n_components=n_components,
                  random_state=random_state)
        if isinstance(dependentVar_cols, list):
            if isinstance(dependentVar_cols[0], str):
                X = df[dependentVar_cols].to_numpy()
                columns = dependentVar_cols
            elif isinstance(dependentVar_cols[0],np.ndarray):
                X = np.concatenate([i.reshape(-1, 1) 
                                    if i.ndim == 1 else i for i in dependentVar_cols], axis=1)
        elif isinstance(dependentVar_cols,np.ndarray):
            X = dependentVar_cols
        elif isinstance(dependentVar_cols,pd.DataFrame):
            X = dependentVar_cols.to_numpy()
            columns = dependentVar_cols.columns.to_list()
        if scaling:
            X = StandardScaler().fit_transform(X)
        X_pca = pca.fit_transform(X)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_matrix = pd.DataFrame(loadings, index = columns)
        return pca, X_pca, loading_matrix
        
    

    
    
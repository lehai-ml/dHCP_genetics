#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:00:18 2022

@author: lh20
"""
from typing import List,Union
import data_exploration
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score


class Metrics:
    
    @staticmethod
    
    

class Linearmodels:
    
    @staticmethod
    

class Nestedcv:
    
    @staticmethod
    def perform_nested_cv(X:Union[pd.DataFrame,np.ndarray],
                          y:Union[pd.DataFrame,np.ndarray],
                          covariates:np.ndarray,
                          model,
                          k=5,
                          **kwargs):
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42
        inner_cv=KFold(n_splits=k,random_state=kwargs['random_state'])
        outer_cv=KFold(n_splits=k,random_state=kwargs['random_state'])
        for trainval_index,test_index in outer_cv.split(X):
            X_trainval=X[trainval_index,:]
            y_trainval=y[trainval_index]
            X_test=X[test_index,:]
            y_test=y[test_index]
            
            # do model selection
            # report the training score, CV score and model score
            
            yield (X_trainval,y_trainval,X_test,y_test)
    

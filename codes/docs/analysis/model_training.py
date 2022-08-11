#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:00:18 2022

@author: lh20
"""
from typing import List,Union,Optional
from collections import defaultdict
try:
    import data_exploration
except ModuleNotFoundError:
    from . import data_exploration
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
import statsmodels.api as sm
from sklearn.decomposition import PCA


class Metrics:
    
    @staticmethod
    def score(y_true:np.ndarray,
              y_pred:np.ndarray,
              scoring:str,prefix:str=None):
        """
        Perform scikitlearn scoring

        Parameters
        ----------
        y_true : np.ndarray
            Array of the observed values
        y_pred : np.ndarray
            Arrays of the predicted values
        scoring : str
            The scoring system. {'r2','neg_mean_squared_error'}
        prefix : str, optional
            If you want to add a prefix to the key of the score dictionary. The default is None.

        Returns
        -------
        score_dict : dict
            The score dictionary. The key is the type of scoring and value is the score

        """
        score_dict = defaultdict()
        if isinstance(scoring, str):
            scoring=[scoring]
        if prefix is None:
            prefix = ''
        for score in scoring:
            if score == 'r2':
                score_dict[score] = r2_score(y_true, y_pred)
            if score == 'neg_mean_squared_error':
                score_dict[score] = -mean_squared_error(y_true,y_pred)
        return score_dict
    
    @staticmethod
    def cross_validate(model,
                       X:np.ndarray,
                       y:np.ndarray,
                       cv:StratifiedKFold,
                       scoring:str,
                       return_train_score:bool=False,
                       bins:int=4):
        """
        Perform cross validation, with added bonus of stratifying the continuous label

        Parameters
        ----------
        model :
            The scikit learn model.
        X : np.ndarray
            the independent Variable.
        y : np.ndarray
            the dependent Variable.
        cv : StratifiedKFold
            the StratifiedKFold model.
        scoring : str
            The scoring system {'r2','neg_mean_squared_error'}.
        return_train_score : bool, optional
            If you want to return the train score. The default is False.
        bins : int, optional
            The bins to divide the continuous label. The default is 4.

        Returns
        -------
        model_summary : dict
            The model summary of each cross-validation score.

        """
        model_summary = defaultdict(dict)
        target_bins = pd.cut(y,bins=bins,labels=False)
        if return_train_score:
            model_summary['train_score'] = defaultdict(list)
        model_summary['test_score'] = defaultdict(list)
        if isinstance(scoring,str):
            scoring = [scoring]
            
        for split_no,(train,test) in enumerate(cv.split(X,target_bins)):
            X_train=X[train,:]
            y_train=y[train].ravel()
            X_test=X[test,:]
            y_test=y[test].ravel()
            model.fit(X_train,y_train)
            for score in scoring:
                if return_train_score:
                    model_summary['train_score'][score].append(Metrics.score(y_train,model.predict(X_train),scoring=score)[score])
                model_summary['test_score'][score].append(Metrics.score(y_test,model.predict(X_test),scoring=score)[score])
        
        return model_summary
    
class PCA_adjuster(BaseEstimator,TransformerMixin):
    
    def __init__(self,variables_to_reduce_by_PCA_idx:Union[List[int],List[str]]=None,
                  list_of_vars:List[str]=None,
                 n_components:Union[float,int]=None):
        
        if variables_to_reduce_by_PCA_idx is not None:
            if isinstance(variables_to_reduce_by_PCA_idx,str):
                variables_to_reduce_by_PCA_idx = [variables_to_reduce_by_PCA_idx]
            if isinstance(variables_to_reduce_by_PCA_idx[0], str):
                if list_of_vars is None:
                    raise KeyError('list of vars is missing')
                variables_to_reduce_by_PCA_idx = [idx for idx,label in enumerate(list_of_vars) if label in variables_to_reduce_by_PCA_idx]
        self.variables_to_reduce_by_PCA_idx=variables_to_reduce_by_PCA_idx        
        self.list_of_vars = list_of_vars
        self.n_components = n_components

    def fit(self,X:np.ndarray,y:np.ndarray=None):
        self.covariates_not_reduced_by_PCA_idx = [i for i in range(X.shape[1]) if i not in self.variables_to_reduce_by_PCA_idx]
        self.pca = PCA(n_components=self.n_components).fit(X[:,self.variables_to_reduce_by_PCA_idx])
        self.X_pca = self.pca.transform(X[:,self.variables_to_reduce_by_PCA_idx])
        # self.model,mass_univariate = data_exploration.MassUnivariate.mass_univariate(cont_independentVar_cols=np.hstack([X[:,covariates_not_reduced_by_PCA_idx],
        #                                                                                     y.reshape(-1,1)]),
        #                                                 dependentVar_cols=self.X_pca)
        self.covariates = X[:,self.covariates_not_reduced_by_PCA_idx]
        return self 
    def transform(self,X:np.ndarray,y:np.ndarray=None):
        new_X_pca = self.pca.transform(X[:,self.variables_to_reduce_by_PCA_idx])
        new_X = np.hstack([new_X_pca,X[:,self.covariates_not_reduced_by_PCA_idx]])
        return new_X
        
        

class AdjustingScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self,cat_independentVar_idx:Union[List[int],List[str]]=None,
                 cont_independentVar_idx:Union[List[int],List[str]]=None,
                 list_of_vars:List[str]=None):
        """
        Used to ajdust variables.
        Give X and y, where X is the ndarray of variables needs to be adjusted and covariates.
        y is ignored.
        
        Parameters
        ----------
        cat_independentVar_idx : [List[int]], optional
            The column in X where it is the categorical covariate. The default is None.
        cont_independentVar_idx : [List[int]], optional
            The column in X where it is the continuous covariate. The default is None.
        list of vars:
            must corresponds to X.shape
        Returns
        -------
        None.

        """
        if cat_independentVar_idx is not None:
            if isinstance(cat_independentVar_idx,str):
                cat_independentVar_idx = [cat_independentVar_idx]
            if isinstance(cat_independentVar_idx[0], str):
                if list_of_vars is None:
                    raise KeyError('list of vars is missing')
                cat_independentVar_idx = [idx for idx,label in enumerate(list_of_vars) if label in cat_independentVar_idx]
        self.cat_independentVar_idx=cat_independentVar_idx
        
        if cont_independentVar_idx is not None:
            if isinstance(cont_independentVar_idx,str):
                cont_independentVar_idx = [cont_independentVar_idx]
            if isinstance(cont_independentVar_idx[0], str):
                if list_of_vars is None:
                    raise KeyError('list of vars is missing')
                cont_independentVar_idx = [idx for idx,label in enumerate(list_of_vars) if label in cont_independentVar_idx]
        
        self.cont_independentVar_idx=cont_independentVar_idx
        self.list_of_vars = list_of_vars
        
    def fit(self,X:np.ndarray,
            y:np.ndarray=None):
        """
        Used to ajdust variables.
        Give X and y, where X is the ndarray of variables needs to be adjusted and covariates.
        y is ignored.
        
        Parameters
        ----------
        cat_independentVar_idx : [List[int]], optional
            The column in X where it is the categorical covariate. The default is None.
        cont_independentVar_idx : [List[int]], optional
            The column in X where it is the continuous covariate. The default is None.

        Returns
        -------
        None.

        """
        self.dependentVar_cols = [i for i in range(X.shape[1]) if i not in (self.cat_independentVar_idx or [])
                             and i not in (self.cont_independentVar_idx or [])]
        self.independentVar_cols = [i for i in range(X.shape[1]) if i not in self.dependentVar_cols]
        if self.cat_independentVar_idx is not None:
            self.cat_independentVar_cols = X[:,self.cat_independentVar_idx]
            if len(np.unique(self.cat_independentVar_cols)) > 2:
                raise KeyError('this only works with 2 categories, e.g. male or females')
        else:
            self.cat_independentVar_cols = None
        if self.cont_independentVar_idx is not None:
            self.cont_independentVar_cols = X[:,self.cont_independentVar_idx]
        else:
            self.cont_independentVar_cols = None
        self.dependentVar = X[:,self.dependentVar_cols]
        
        # get the model list
        _,self.model_list = data_exploration.MassUnivariate.adjust_covariates_with_lin_reg(cat_independentVar_cols=self.cat_independentVar_cols,
                                                                       cont_independentVar_cols=self.cont_independentVar_cols,
                                                                       dependentVar_cols=self.dependentVar,
                                                                       return_model_adjuster=True,
                                                                       scaling=None)
        return self
    def transform(self,X,y=None):
        adjusted_X = np.empty(X.shape)      
        adjusted_X = adjusted_X[:,self.dependentVar_cols]
        for idx in self.model_list.keys():
            X_pred = self.model_list[idx].predict(sm.add_constant(X[:,self.independentVar_cols]))
            adjusted_X[:,idx] = X[:,idx] - X_pred
        return adjusted_X
  
class NestedCV:
    
    @staticmethod
    def perform_nestedcv(model,
                         df: Optional[pd.DataFrame] = None,
                         cat_independentVar_cols: Union[List[str],
                                                        np.ndarray,
                                                        pd.DataFrame,
                                                        pd.Series] = None,
                         cont_independentVar_cols: Union[List[str],
                                                        np.ndarray,
                                                        pd.DataFrame,
                                                        pd.Series] = None,
                         dependentVar_cols: Union[List[str],
                                                        np.ndarray,
                                                        pd.DataFrame,
                                                        pd.Series] = None,
                         n_splits:int = 5,
                         bins:int = 4,
                         scaling:str='x',
                         drop_const:bool=True,
                         grid_searchCV:bool=False,
                         scoring:str='neg_mean_squared_error'):
        
        model_summary = defaultdict(dict)
        
        dependentVar, independentVar = data_exploration.MassUnivariate.prepare_data(df = df,
                                                                   cat_independentVar_cols=cat_independentVar_cols,
                                                                   cont_independentVar_cols=cont_independentVar_cols,
                                                                   dependentVar_cols=dependentVar_cols,
                                                                   scaling=None)
        if drop_const:
            independentVar = independentVar.drop(columns=['const'])
        
        dependent_Var_names = dependentVar.columns
        independent_Var_names = independentVar.columns
        
        inner_cv = StratifiedKFold(n_splits=n_splits)
        outer_cv = StratifiedKFold(n_splits=n_splits)
        
        X = independentVar.to_numpy()
        y = dependentVar.to_numpy()
        
        
        
        for idx,target in enumerate(dependent_Var_names):
            model_summary[target] = defaultdict(dict)
            target_bins = pd.cut(y[:,idx],bins=bins,labels=False)
            
            for split_no,(trainval_index,test_index) in enumerate(outer_cv.split(X,target_bins)):
                X_trainval=X[trainval_index,:]
                y_trainval=y[trainval_index,idx].ravel()
                X_test=X[test_index,:]
                y_test=y[test_index,idx].ravel()

                # do model selection
                scores = Metrics.cross_validate(model,X_trainval,y_trainval,bins=bins,cv = inner_cv,scoring=scoring,return_train_score=True)
                estimator = model.fit(X_trainval,y_trainval)
                model_summary[target][f'split_{split_no}'][f'cv_{scoring}'] = scores['test_score'][scoring]
                model_summary[target][f'split_{split_no}']['y_pred'] = estimator.predict(X_test)
                model_summary[target][f'split_{split_no}']['y_test'] = y_test
                model_summary[target][f'split_{split_no}']['test_r2_score'] = r2_score(y_test, estimator.predict(X_test))
                model_summary[target][f'split_{split_no}']['estimator'] = estimator

        return model_summary
    
    
    
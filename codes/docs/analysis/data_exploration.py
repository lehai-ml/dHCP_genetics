"""
data_exploration.py
Uses to perform mass univariate and everything related to it
"""
from typing import List, Union, Optional
from collections import defaultdict
from itertools import combinations
import statsmodels.api as sm
from scipy.stats import ttest_ind, zscore
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tqdm
# import pickle
from statsmodels.multivariate.cancorr import CanCorr


class MassUnivariate:
    
    @staticmethod
    def mass_univariate(df: Optional[pd.DataFrame] = None,
                        cat_independentVar_cols: Optional[Union[List[str],
                                                        List[np.ndarray]]] = None,
                        cont_independentVar_cols: Optional[Union[List[str],
                                                                List[np.ndarray]]] = None,
                        dependentVar_cols: Optional[Union[List[str],
                                                        List[np.ndarray]]] = None,
                        scaling: str='both',
                        col_to_drop : Optional[List[str]] = None,
                        additional_info=None) -> Union[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
        """[Returns the model and model summary
            Performs univariate test, implements statsmodel.api.OLS]

        Args:
            df (Optional[pd.DataFrame]): [pandas data frame, where each row is one observation]
            cat_independentVar_cols (Optional[Union[List[str], List[np.ndarray]]], optional): [list of categorical variables, such that they will be converted to dummies by pandas]. Defaults to None.
            cont_independentVar_cols (Optional[Union[List[str], List[np.ndarray]]], optional): [list of continuous variables, they will be appended to cat_independentVar_cols]. Defaults to None.
            dependentVar_cols (Optional[Union[List[str], List[np.ndarray]]], optional): [the dependent variable, the ]. Defaults to None.
            scaling [str]: Default = both. ('x','y','both','none')
            If none, we will not perform standard scaler, 'both' we will perform on both x (independent variable- feature) and y (dependent variable- target)
            col_to_drop [list[str]] = if we want to remove any column from the independent Variable before fitting the model.
            additional_info ([type], optional): [additional columns to be appended]. Defaults to None.

        Returns:
            Union[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]: [the statsmodel model and dataframe of the model summary, where cols are independent variables and const]
        """
        model_summary = defaultdict(list)
        if type(df) == pd.DataFrame:
            new_df = df.copy()

        cat_independentVar = defaultdict(list)
        cont_independentVar = defaultdict(list)
        dependentVar = defaultdict(list)
        independentVar = dict()
        if type(cat_independentVar_cols) == list:
            for idx, cat_independentVar_col in enumerate(cat_independentVar_cols):
                if type(cat_independentVar_col) == str:
                    cat_independentVar_temp = pd.get_dummies(new_df[cat_independentVar_col].values,
                                                            prefix=cat_independentVar_col,
                                                            drop_first=True).to_dict(orient='list')

                    cat_independentVar.update(cat_independentVar_temp)
                if type(cat_independentVar_col) == np.ndarray:
                    cat_independentVar_temp = pd.get_dummies(cat_independentVar_col,
                                                            prefix='Cat_' +
                                                            str(idx),
                                                            drop_first=True).to_dict(orient='list')

                    cat_independentVar.update(cat_independentVar_temp)
        elif type(cat_independentVar_cols) == np.ndarray:
            if cat_independentVar_cols.ndim == 1:
                cat_independentVar_temp = pd.get_dummies(cat_independentVar_cols,
                                                        prefix='Cat_',
                                                        drop_first=True).to_dict(orient='list')
                cat_independentVar.update(cat_independentVar_temp)
            else:
                for col in range(cat_independentVar_cols.shape[1]):
                    cat_independentVar_temp = pd.get_dummies(cat_independentVar_cols[:, col],
                                                            prefix='Cat_'+col+'_',
                                                            drop_first=True).to_dict(orient='list')
                    cat_independentVar.update(cat_independentVar_temp)

        if type(cont_independentVar_cols) == list:
            if not cont_independentVar_cols:
                pass
            elif type(cont_independentVar_cols[0]) == str:
                cont_independentVar_temp = np.asarray(
                    new_df.loc[:, cont_independentVar_cols])
                if cont_independentVar_temp.ndim == 1:
                    cont_independentVar_temp = cont_independentVar.reshape(-1, 1)
                if scaling=='both' or scaling == 'x':
                    cont_independentVar_temp = StandardScaler().fit_transform(cont_independentVar_temp)
                cont_independentVar_temp = pd.DataFrame(cont_independentVar_temp)
                cont_independentVar_temp.columns = cont_independentVar_cols
                cont_independentVar = cont_independentVar_temp.to_dict(
                    orient='list')

            elif type(cont_independentVar_cols[0]) == np.ndarray:
                cont_independentVar_temp = np.hstack(cont_independentVar_cols)
                if cont_independentVar_temp.ndim == 1:
                    cont_independentVar_temp = cont_independentVar.reshape(-1, 1)
                if scaling=='both' or scaling == 'x':
                    cont_independentVar_temp = StandardScaler().fit_transform(cont_independentVar_temp)
                cont_independentVar_temp = pd.DataFrame(cont_independentVar_temp)
                cont_independentVar_temp.columns = [
                    'Cont_'+str(i) for i in range(cont_independentVar_cols.shape[1])]
                cont_independentVar = cont_independentVar_temp.to_dict(
                    orient='list')

        elif type(cont_independentVar_cols) == np.ndarray:
            cont_independentVar_temp = cont_independentVar_cols
            if cont_independentVar_temp.ndim == 1:
                cont_independentVar_temp = cont_independentVar_temp.reshape(-1, 1)
            if scaling=='both' or scaling == 'x':
                cont_independentVar_temp = StandardScaler().fit_transform(cont_independentVar_temp)
            cont_independentVar_temp = pd.DataFrame(cont_independentVar_temp)
            cont_independentVar_temp.columns = [
                'Cont_'+str(i) for i in range(cont_independentVar_temp.shape[1])]
            cont_independentVar = cont_independentVar_temp.to_dict(orient='list')

        if type(dependentVar_cols) == list:

            if type(dependentVar_cols[0]) == str:
                dependentVar_temp = np.asarray(new_df.loc[:, dependentVar_cols])
                if dependentVar_temp.ndim == 1:
                    dependentVar_temp = dependentVar_temp.reshape(-1, 1)
                if scaling=='both' or scaling == 'y':
                    dependentVar_temp = StandardScaler().fit_transform(dependentVar_temp)
                dependentVar_temp = pd.DataFrame(dependentVar_temp)
                dependentVar_temp.columns = dependentVar_cols
                dependentVar = dependentVar_temp.to_dict(orient='list')
            elif dependentVar_cols[0] == np.ndarray:
                dependentVar_temp = np.hstack(dependentVar_cols)
                if dependentVar_temp.ndim == 1:
                    dependentVar_temp = dependentVar_temp.reshape(-1, 1)
                if scaling=='both' or scaling == 'y':
                    dependentVar_temp = StandardScaler().fit_transform(dependentVar_temp)
                dependentVar_temp = pd.DataFrame(dependentVar_temp)
                dependentVar_temp.columns = [
                    'Dependent_Var_'+str(i) for i in range(dependentVar_cols.shape[1])]
                dependentVar = dependentVar_temp.to_dict(orient='list')
        elif type(dependentVar_cols) == np.ndarray:
            dependentVar_temp = dependentVar_cols
            if dependentVar_temp.ndim == 1:
                dependentVar_temp = dependentVar_temp.reshape(-1, 1)
            if scaling=='both' or scaling == 'y':
                dependentVar_temp = StandardScaler().fit_transform(dependentVar_temp)
            dependentVar_temp = pd.DataFrame(dependentVar_temp)
            dependentVar_temp.columns = [
                'Dependent_Var_'+str(i) for i in range(dependentVar_cols.shape[1])]
            dependentVar = dependentVar_temp.to_dict(orient='list')

        independentVar.update(cont_independentVar)
        independentVar.update(cat_independentVar)
        if not independentVar:
            independentVar['const'] = [1. for i in range(
                len(list(dependentVar.values())[0]))]
            independentVar = pd.DataFrame(independentVar)
        else:
            independentVar = pd.DataFrame(independentVar)
            independentVar = sm.add_constant(independentVar)
        
        if col_to_drop:
            try:
                independentVar = independentVar.drop(col_to_drop,axis=1)
            except KeyError:
                # print('You might be trying to remove the categorical data, which in this function is written as Gender_2.0 or sth.')
                independentVar = independentVar.drop([i for i in col_to_drop if i in independentVar.columns],axis=1)
        dependentVar = pd.DataFrame(dependentVar)

        for feature in dependentVar.columns:
            last_model = sm.OLS(dependentVar.loc[:, feature], independentVar).fit()
            result = [None] * (len(last_model.params) + len(last_model.pvalues))
            result[::2] = last_model.params
            result[1::2] = last_model.pvalues
            model_summary[feature].extend(result)

        model_summary = pd.DataFrame(model_summary).T
        list1 = independentVar.columns.to_list()
        list2 = ['_coef', '_pval']

        model_summary.columns = [i + n for i in list1 for n in list2]
        if additional_info:
            model_summary['Additional_info'] = additional_info
            model_summary.columns = [i + n for i in list1
                                    for n in list2] + ['Additional_info']

        return last_model, model_summary
    
    @staticmethod
    def remove_outliers(df:pd.DataFrame,
                        col:Union[str,List]=None,
                        threshold:float=None,
                        remove_schemes:str='any',
                        percentage_of_outlier:float=0.2,
                        subject_ID:list=None) -> pd.DataFrame:
        """[Remove outliers using the z-score (the same as using StandardScaler). The standard deviation]

        Args:
            df (pd.DataFrame): [Dataframe of interest]
            col (str or List, optional): [column of interest]. Defaults to None.
            threshold (float, optional): [the threshold of the z-score]. Defaults to None.
            subject_ID (list, optional): [Or gives me the list of subject IDs and remove by that way]. Defaults to None.

        Returns:
            pd.DataFrame: [New data frame without the outliers]
        """
        new_df = df.copy()
        idx_to_remove_by_col=[]
        idx_to_remove_by_ID=[]
        if col is not None:
            if not isinstance(col,list):
                col = [col]
            if remove_schemes == 'any': #remove observations that have outlier in any of the examined column
                idx_to_remove_by_col = new_df.loc[(np.abs(zscore(new_df[col]))>threshold).any(axis=1)].index.to_list()
            elif remove_schemes == 'all': #remove observations that have outlier in all of the examined column
                idx_to_remove_by_col = new_df.loc[(np.abs(zscore(new_df[col]))>threshold).all(axis=1)].index.to_list()
            elif remove_schemes == 'sum': # sum up the cols of interests and remove the outlier based on that sum
                idx_to_remove_by_col = new_df.loc[(np.abs(zscore(new_df[col].sum(axis=1)))>threshold)].index.to_list()
            elif remove_schemes == 'percentage': # remove observations that have more than x% of all features are outliers
                all_zscores = np.abs(zscore(new_df[col]))
                idx_to_remove_by_col = new_df.loc[(all_zscores>threshold).mean(axis=1)>=percentage_of_outlier].index.to_list()
        if subject_ID is not None:
            idx_to_remove_by_ID = (new_df.iloc[[idx for idx,i in enumerate(new_df.ID) if i in subject_ID]].index.to_list())
        idx_to_remove = idx_to_remove_by_col + idx_to_remove_by_ID
        idx_to_remove = list(set(idx_to_remove))
        new_df = new_df.drop(idx_to_remove).reset_index(drop = True)
        return new_df

    @staticmethod
    def adjust_covariates_with_lin_reg(y: Union[np.ndarray, pd.DataFrame, pd.Series], *covariates: Union[List[np.ndarray], pd.DataFrame, pd.Series]) -> np.ndarray:
        """[Adjusting for covariates with linear regression; take the residual after fitting to the linear regression model]

        Args:
            y (np.ndarray): [the variable to be adjusted]
            *covariates (List[np.ndarray]): list of variables
        Returns:
            np.ndarray: [residual]

        Note:
            Linear Regression Sklearn and sm.OLS gives the same residual and coefs.
            Here, Standard Scaler is applied to the independent Variables.
        """
        X = np.concatenate([i.reshape(-1, 1) if i.ndim ==
                        1 else i for i in covariates], axis=1)
        X = StandardScaler().fit_transform(X)
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        y_pred = lin_reg.predict(X)
        new_feature = y-y_pred
        return new_feature
    
    @classmethod
    def calculate_mass_univariate_across_multiple_thresholds(
            cls,
            df: pd.DataFrame,
            thresholds: List[str],
            cat_independentVar_cols: Optional[Union[List[str],
                                            List[np.ndarray]]] = None,
            cont_independentVar_cols: Optional[Union[List[str],
                                                    List[np.ndarray]]] = None,
            dependentVar_cols: Optional[Union[List[str],
                                            List[np.ndarray]]] = None,**kwargs) -> pd.DataFrame:
        """
        

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe.
        thresholds : List[str]
            DESCRIPTION.
        see MassUnivariate.mass_univariate for full variable descriptions
        Returns
        -------
        new_df : TYPE
            DESCRIPTION.

        """
        new_df = pd.DataFrame()
        for threshold in tqdm.tqdm(thresholds):
            _, temp_model_summary = cls.mass_univariate(
                df,
                cat_independentVar_cols=cat_independentVar_cols,
                cont_independentVar_cols=cont_independentVar_cols + [threshold],
                dependentVar_cols=dependentVar_cols,**kwargs)
    
            temp_model_summary.reset_index(drop=False, inplace=True)
            temp_model_summary.rename(
                {
                    'index': 'Connection',
                    threshold + '_coef': 'PRS_coef',
                    threshold + '_pval': 'PRS_pval'
                },
                axis=1,
                inplace=True)
            temp_model_summary['threshold'] = threshold
            new_df = pd.concat([new_df,temp_model_summary],axis=0)
    
        new_df.reset_index(drop=True, inplace=True)
        return new_df

    @classmethod
    def calculate_R_squared_explained(cls,df: pd.DataFrame,
                                      col_to_drop:List[str]=None,**kwargs) -> float:
        """[Calculate R squared explained from the model]

        Args:
            df ([pd.DataFrame]): [the model to be passed to the univariate model]
            col_to_drop (List[str], optional): [the column names of the data frame to be removed in the univariate model]. Defaults to None.
            **kwargs:
            cat_independentVar_cols (Optional[Union[List[str], List[np.ndarray]]], optional): [list of categorical variables, such that they will be converted to dummies by pandas]. Defaults to None.
            cont_independentVar_cols (Optional[Union[List[str], List[np.ndarray]]], optional): [list of continuous variables, they will be appended to cat_independentVar_cols]. Defaults to None.
            dependentVar_cols (Optional[Union[List[str], List[np.ndarray]]], optional): [the dependent variable, the ]. Defaults to None.
            scaling [str]: Default = both. ('x','y','both','none')
            If none, we will not perform standard scaler, 'both' we will perform on both x (independent variable- feature) and y (dependent variable- target)            col_to_drop [list[str]] = if we want to remove any column from the independent Variable before fitting the model.
            additional_info ([type], optional): [additional columns to be appended]. Defaults to None.
        Returns:
            [float]: [difference in the full and the null model R squared]
        
        """
        full_model, _ = cls.mass_univariate(df,**kwargs)
        if not col_to_drop:
            return full_model.rsquared
        null_model, _ = cls.mass_univariate(df,col_to_drop=col_to_drop, **kwargs)
        return full_model.rsquared - null_model.rsquared
        
    @classmethod
    def check_all_predictors_combo_linear_Reg(cls,df: Optional[pd.DataFrame],
                                            cat_independentVar_cols: Optional[List[str]] = None,
                                            cont_independentVar_cols: Optional[List[str]] = None,
                                            dependentVar_cols: Optional[List[str]] = None):
        """[Perform univariate test on all combinations of predictors]

        Args:
            df (Optional[pd.DataFrame]): [description]
            cat_independentVar_cols (Optional[List[str]], optional): [description]. Defaults to None.
            cont_independentVar_cols (Optional[List[str]], optional): [description]. Defaults to None.
            dependentVar_cols (Optional[List[str]], optional): [description]. Defaults to None.
        Returns:
            [type]: [description]
        """
        all_models = defaultdict(list)

        def print_model_results(model: sm.regression.linear_model.RegressionResultsWrapper,
                                model_name: str,
                                dictionary: dict = all_models) -> None:
            variables = model.params.index.to_list()
            len_variables = len(variables)-1
            coefs = model.params.to_list()
            p_vals = model.pvalues.to_list()
            aic = model.aic
            R2 = model.rsquared
            R2_adj = model.rsquared_adj

            dictionary[model_name].append(
                [len_variables, variables, coefs, p_vals, aic, R2, R2_adj])

        null_model, _ = cls.mass_univariate(df=df,
                                        cat_independentVar_cols=None,
                                        cont_independentVar_cols=None,
                                        dependentVar_cols=dependentVar_cols)
        print_model_results(null_model, '0_0')
        if not cat_independentVar_cols:
            cat_independentVar_cols = []
        if not cont_independentVar_cols:
            cont_independentVar_cols = []
        total_predictors = cat_independentVar_cols + cont_independentVar_cols

        # for k in range(1,len(total_predictors)+1):
        for k in tqdm.tqdm(range(1, len(total_predictors)+1)):
            k_combo = combinations(total_predictors, k)
            for idx, model_combo in enumerate(k_combo):

                cat_independentVar_cols_temp = []
                cont_independentVar_cols_temp = []
                for _, covar in enumerate(model_combo):

                    if covar in cat_independentVar_cols:
                        cat_independentVar_cols_temp.append(covar)
                    elif covar in cont_independentVar_cols:
                        cont_independentVar_cols_temp.append(covar)

                model_temp, _ = cls.mass_univariate(df=df,
                                                cat_independentVar_cols=cat_independentVar_cols_temp,
                                                cont_independentVar_cols=cont_independentVar_cols_temp,
                                                dependentVar_cols=dependentVar_cols)

                model_name_temp = str(k)+'_'+str(idx)
                print_model_results(model_temp, model_name_temp)

        return all_models

    @classmethod
    def preprocess_forward_selection(cls,dictionary: dict) -> List[pd.DataFrame]:
        """[Preprocess the info of mass univariate tests, all possible combo of predictors
        Input is the output from ```check_all_predictors_combo_linear_Reg```]

        Args:
            dictionary (dict): [Output form ```check_all_predictors_combo_linear_Reg```]

        Returns:
            List[pd.DataFrame]: [4 data frames - score_summary, variable combo summary, beta coefs, p-values]
        """
        temp_df = pd.DataFrame(dictionary).T.reset_index(drop=True)
        temp_df = pd.concat([temp_df.drop(0, axis=1), temp_df[0].apply(
            pd.Series)], axis=1)  # this will preprocess the dictionary
        temp_df.columns = ['N_var', 'Var', 'Beta', 'P_val', 'AIC', 'R2', 'R2_adj']
        model_score_summary = temp_df[['N_var', 'AIC', 'R2', 'R2_adj']].copy()
        model_var_summary = temp_df[['Var']].copy()
        all_predictors = model_var_summary.iloc[-1, 0]
        label_binarizer = LabelBinarizer().fit(all_predictors)
        model_var_summary = pd.DataFrame(model_var_summary['Var'].apply(
            lambda x: label_binarizer.transform(x).sum(axis=0)))['Var'].apply(pd.Series)
        model_var_summary.columns = label_binarizer.classes_
        model_var_summary = model_var_summary[all_predictors]

        model_beta_summary = temp_df['Beta']
        model_p_summary = temp_df['P_val']
        model_beta = defaultdict(list)
        model_p = defaultdict(list)
        for idx in range(len(model_var_summary)):
            beta_iter = iter(model_beta_summary.iloc[idx])
            p_iter = iter(model_p_summary.iloc[idx])
            for n in model_var_summary.iloc[idx]:
                if n == 1:
                    model_beta[idx].append(next(beta_iter))
                    model_p[idx].append(next(p_iter))
                else:
                    model_beta[idx].append(np.nan)
                    model_p[idx].append(np.nan)
        model_beta_summary = pd.DataFrame(model_beta).T
        model_p_summary = pd.DataFrame(model_p).T
        model_beta_summary.columns = model_var_summary.columns
        model_p_summary.columns = model_var_summary.columns

        return model_score_summary, model_var_summary, model_beta_summary, model_p_summary

    @classmethod
    def print_summary_table(cls,
                            df:pd.DataFrame,
                            thresholds:List[str]=None,
                            cat_independentVar_cols:List[str]=None,
                            cont_independentVar_cols:List[str]=None,
                            dependentVar_cols:List[str]=None)->pd.DataFrame:
        """
        

        Parameters
        ----------
        see data_exploration.MassUnivariate.mass_univariate for full description of the variables
        Returns
        -------
        summary_table : pd.DataFrame
            The summary table containing R, Beta value, P-value for each dependent variable at each PRS threshold.

        """
        Result_dict=defaultdict(dict)
        for dependent_variable in tqdm.tqdm(dependentVar_cols):
            Result_dict[dependent_variable] = defaultdict(dict)
            for threshold in thresholds:
                Result_dict[dependent_variable][threshold] = defaultdict(dict)
                Result_dict[dependent_variable][threshold]['R2'] = defaultdict(dict)
                temp_model, _ = cls.mass_univariate(
                    df=df,
                    cat_independentVar_cols=cat_independentVar_cols,
                    cont_independentVar_cols=cont_independentVar_cols + [threshold],
                    dependentVar_cols=[dependent_variable]) # calculate a full model
                Result_dict[dependent_variable][threshold]['Beta'] = temp_model.params[1:].to_dict()
                Result_dict[dependent_variable][threshold]['P_val'] = temp_model.pvalues[1:].to_dict()
                for col_to_drop in cont_independentVar_cols + cat_independentVar_cols + [threshold]:
                    Result_dict[dependent_variable][threshold]['R2'][
                        col_to_drop] = cls.calculate_R_squared_explained(
                            df=df,
                            col_to_drop=col_to_drop,
                            cat_independentVar_cols=cat_independentVar_cols,
                            cont_independentVar_cols=cont_independentVar_cols,
                            dependentVar_cols=[dependent_variable])
        
        summary_table = pd.DataFrame.from_dict({(i,j,k): Result_dict[i][j][k] 
                                                for i in Result_dict.keys() 
                                                for j in Result_dict[i].keys() 
                                                for k in Result_dict[i][j].keys()},
                       orient='index')
        
        return summary_table
        
    
def matSpDLite(correlation_matrix:np.ndarray,alpha:str = 0.05):
    """
    Adapted from Nyholt DR R script.
    Please cite
    Nyholt DR (2004) A simple correction for multiple testing for SNPs in linkage disequilibrium with each other. Am J Hum Genet 74(4):765-769.
    and 
    http://gump.qimr.edu.au/general/daleN/matSpDlite/
    and
    Li and Ji 2005. 
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        The correlation matrix. It must be symmetric.
    alpha : str, optional
        DESCRIPTION. The default is 0.05.

    """
    evals,_ = np.linalg.eigh(correlation_matrix.T)
    evals = np.sort(evals)[::-1]
    
    oldV = np.var(evals,ddof=1)
    M = len(evals)
    L = M-1
    Meffold = (L*(1- oldV/M)) + 1
    
    print(f'Effective Number of Independent Variables [Veff] is {Meffold}')
    newevals = np.where(evals<0,0,evals) # negative eigenvalues are set to 0
    
    IntLinewevals = np.where(newevals>=1,1,0) #the integral part "represents identical tests that should be counted as one in Meff"
    NonIntLinewevals = newevals - np.floor(newevals) # the non integral part should represent the partially correlated test that should be counted as a fractional number between 0 and 1
    MeffLi = np.sum(IntLinewevals + NonIntLinewevals)
    
    print(f'Effective Number of Independent Variables [VeffLi] (Using equation 5 of Li and Ji 2005) is {np.round(MeffLi)}')
    
    if MeffLi < Meffold:
        adjusted_p_val = alpha/MeffLi
    else:
        adjusted_p_val = alpha/Meffold
    print(f'The adjusted multiple testing correction p-val is alpha/lower(Meff) = {adjusted_p_val}')



# def save_dict_with_pickle(dictionary: Optional[dict],
#                           file_path: Optional[str]) -> None:
#     a_file = open(file_path, 'wb')
#     pickle.dump(dictionary, a_file)
#     a_file.close()


# def open_dict_with_pickle(file_path: Optional[str]) -> dict:
#     a_file = open(file_path, 'rb')
#     dictionary = pickle.load(a_file)
#     return dictionary

def perform_CCA(X,Y,scale=True):
    if scale:
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)
    
    stats_cca = CanCorr(X, Y)
    return stats_cca.corr_test().summary()

class Stability_tests:
    
    @staticmethod
    
    def train_test_split_modified(df: Union[pd.DataFrame, np.ndarray] = None,*stratify_by_args, bins : int =4,
                                  test_size: float = 0.5, random_state: int = 42) -> tuple:
        """[splitting the data based on several stratification]
    
        Args:
            df (Union[pd.DataFrame, np.ndarray], optional): [the data/ dataframe to be split]. Defaults to None.
            bins (int, optional): [the bins used to separate the continuous data,
                                        if it is too high, then due to small dataset, there might be error due to not enough data in each class]. Defaults to 4.
            This works by concatenate multiple stratification criteria together, and then attempt train test split. This is done at each level of stratification. If the stratificatio is not successful, we reduce the bins number, and try again.
            This is done in succession, so depending on what you want to stratify by, you put it earlier in the list. so GA_vol, PMA_vol will stratify first by GA, and then PMA.
        Returns:
            tuple : train and test set.
        """
        def attempting_train_test_split(df, stratify_by, idx, test_size,random_state):
            try:
                train, test = train_test_split(df, stratify=stratify_by, test_size=test_size, random_state=random_state)
                return train, test
            except ValueError:
                stratification_list.pop()  # remove the last iteration
                print('changing bins for %d argument' % idx)
    
        input_bins = bins
        stratification_list = []
        for idx, stratification in enumerate(stratify_by_args):
            bins = input_bins
            while True:
                if isinstance(stratification, str):
                    strat = df.loc[:, stratification].values
                    if isinstance(strat[0], float):
                        strat = pd.cut(strat, bins=bins, labels=False)
                elif isinstance(stratification, np.ndarray):
                    strat = stratification
                    if isinstance(stratification[0], float):
                        strat = pd.cut(strat, bins=bins, labels=False)
                stratification_list.append(strat)
                stratify_by = [''.join(map(lambda x: str(x), i))
                               for i in zip(*stratification_list)]
                train_test = attempting_train_test_split(df, stratify_by, idx,test_size=test_size,random_state=random_state)
                if train_test:
                    break
                else:
                    bins -= 1

        return train_test
    @staticmethod
    def divide_high_low_risk(y: Optional[np.ndarray], high_perc: float = 0.1, low_perc: float = 0.3) -> List[np.ndarray]:
        """[Divide the dataset into top and bottom x%. Here, we take the assumption that top percent is high risk and bottom percent is low risk.
            While this method maximises the odd ratio for impacts, it raises concerns about the arbitrariness of the quantile used]
    
        Args:
            y (Optional[np.ndarray]): [the raw PRS score]
            high_perc (float, optional): [higher risk group percentage]. Defaults to 0.1.
            low_perc (float, optional): [lower risk group percentage]. Defaults to 0.3.
    
        Raises:
            Exception: [If high risk + low risk > total subject, raise exception]
    
        Returns:
            Union[np.ndarray]: [list of indices]
        """
    
        high_risk_number = int(np.ceil(high_perc*len(y)))
        low_risk_number = int(np.ceil(low_perc*len(y)))
        if high_risk_number+low_risk_number > len(y):
            raise Exception('The high and low risk selection overlapped')
        # bottom
        low_risk = np.argsort(y)[:low_risk_number]
        # top
        high_risk = np.argsort(y)[::-1][:high_risk_number]
    
        return high_risk, low_risk
    
    
    @staticmethod
    def perform_t_test(group_1: np.ndarray,
                       group_2: np.ndarray) -> List[float]:
        """[perform t-test]
    
        Args:
            group_1 (np.ndarray): [The arrays must have the same shape, except in the dimension corresponding to axis]
            group_2 (np.ndarray):
    
        Returns:
            List[float]: [stats,pval]
        """
        stats, pval = ttest_ind(group_1, group_2, equal_var=True)
        return stats, pval


"""
Executing the finetuning and model selection. This is used for regression 
    model, but can be modified to fit other problems.
"""
import warnings
warnings.simplefilter(action='ignore', category=Warning)


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_regression, RFECV, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.inspection import permutation_importance

from sklearn.base import BaseEstimator,TransformerMixin

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import joblib #to save models

import inspect
import preprocessing_step.running_model as running_model

import numpy as np
import sys
    
import pickle
import operator


def save_a_model(model, model_name, split_no, filepath):
    """
    ___________________________
    Save the model externally
    ___________________________
    Args:
        model (scikit-model object)
        model_name(str): name of the model
        split_no(int): split number
        filepath(str): filepath
    
    Return:
        saves the model externally
    """
    filepath = filepath+'split_'+str(split_no)+model_name+'.pkl'
    return joblib.dump(model, filepath)

def save_a_npy(npy_array, npy_name, split_no, filepath):
    """
    ______________________________________________________
    Save the numpy array in binary format externally
    ______________________________________________________
    Args:
        npy_array (numpy array)
        npy_name(str): name of the array
        split_no(int): split number
        filepath(str): filepath
    
    Return:
        saves the array in binary format externally
    """
    filepath = filepath+'split_'+str(split_no)+npy_name+'.npy'
    return np.save(filepath, npy_array)

def save_the_object(object, filepath):
    
    with open(filepath, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

def load_the_object(filepath):
    with open(filepath, 'rb') as input:
        x = pickle.load(input)
    return x

class get_permutation_importances(BaseEstimator,TransformerMixin):
    """
    __________________________________________________________________________
    Get permutation importance for each feature. And then return the ones with 
        the importances above the mean.
    ___________________________________________________________________________
    First a baseline metric is caculated from the data X. Then the feature 
        columns are permuted and the metric is calculated again. The 
        permutation importance is difference between the baseline metrics and 
        metric permutating the feature columns
    
    Args:
        model (scikit model):
        X (np.array):
        y (np.array):
        scoring (str): default 'r2'
        
    Return:
        indices: the indices of the feature importances.
    """
    def __init__(self,model,scoring='r2'):
        self.model=model
        self.scoring=scoring
        
    def fit(self,X,y,n_repeats=10,random_state=42):
        self.model.fit(X,y)
        result = permutation_importance(self.model, X, y, scoring=self.scoring, n_repeats=10, random_state=42)
        mean_threshold = np.mean(result.importances_mean)
        self.indices = np.where(result.importances_mean >= mean_threshold)[0]

        return self
    
    def transform(self,X,y=None):
        X_new=X[:,self.indices]

        return X_new
        


def fine_tune_hyperparameters(param_dict, model, X, y, model_name, fine_tune='grid', cv=4, scoring='r2'):
    """
    ______________________________________________________
    Manual Fine tuning of hyperparameters
    ______________________________________________________
    Args:
        param_dict=dict({'lin_reg':None,
                         'lasso':{'alpha':[0]},
                         'ridge':{'alpha':np.linspace(200,500,10)},
                         'random_forest':{'n_estimators':[3,10,30],
                         'max_features':[2,4,6,8]},
                         'svm':{},
                         'knn':{}})

                         
        param_dict (dict): the parameters to tune for
        model (scikit object): the pipe line that needs to be fine tuned
        X (np.array): the dataset
        y (1D np.array): the label target
        model_name (str): name corresponding to the param_dict.keys()
        fine_tune (str): using either 'grid' or 'randomized'
        cv (KFold or int)= cross-validation partition.
        scoring (str): scoring metric used
        
    
    Return:
        
        search.best_estimator_: best estimator from the fine_tuning process
    """
    examined_param_grid={} #this will be the dictionary that contains the past examined parameter grids
    model_examined=0
    while True:
        if model_examined!=0:
            print('past parameters examined:\n')
            n=[print(i) for i in [(key,value['search_best_params'],value['best_score']) for key,value in examined_param_grid.items()]]
            
        print('make sure you write as if you are writing a command, i.e. with square bracket for list')
        try:
            x=[eval(input(i)) for i in param_dict.keys()]#this will take the input from the user. make sure to type [1,2,3,4] (i.e. with square brackets)
            
        except SyntaxError:
            print('try again')
            continue

        except AttributeError:
            print('There is no parameters to fine_tune, if this is not the case, please add parameters to the model_training.parameters_dict if that is not correct \n')
            model.fit(X,y)
            return model
            break

        param_grid=dict(zip(param_dict.keys(),x))
        model_examined+=1
        examined_param_grid[model_examined]={}#this will create a dynamic nested dicionary
        examined_param_grid[model_examined]['param_grid']=param_grid
        
        if fine_tune=='grid':
            search=GridSearchCV(model,param_grid=param_grid,cv=cv,iid=False,scoring=scoring,verbose=1,n_jobs=-1)
            
        else:
            search=RandomizedSearchCV(model,param_distributions=param_grid,cv=cv,n_iter=20,scoring=scoring,verbose=1,n_jobs=-1)
        
        search.fit(X,y)
        
        examined_param_grid[model_examined]['search_model_best_estimator']=search.best_estimator_# save into that dictionary the best_estimator, best_parameter and best scores.
        examined_param_grid[model_examined]['search_best_params']=search.best_params_
        examined_param_grid[model_examined]['best_score']=search.best_score_

        cvres=search.cv_results_
        
        print('here is the fine tuning results \n')
        for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
            print(mean_score,params)
        
        loop_continue=input('Do you want to repeat? (yes/no): ')
        if loop_continue=='no':
            print('which model you want to choose:')# choose from the list the best model
            n=[print(i) for i in [(key,value['search_best_params'],value['best_score']) for key,value in examined_param_grid.items()]]
            the_model_i_want=eval(input('select a model number'))
            return examined_param_grid[the_model_i_want]['search_model_best_estimator']
            break

def performing_rfecv(model,X,y,combination_idx,split_no,filepath,cv,scoring='r2'):
    """
    ___________________________________________________
    Performing scikit recursive feature elimination CV.
    ___________________________________________________
    Args:
        model (scikit-model): pipe scikit object          
        X (np.asarray): Training dataset
        y (np.asarray): label target
        combination_idx (np.array): feature indices of the X.
        split_no (int):
        filepath (str):
        cv (cross-validated)
        scoring (str)
        
    Returns:
        rfecv fitted model (rfecv scikit object)
        combination_idx_after_rfecv
        scores_after_rfecv
        
    ============================================================================
    Notes:
    
    Here, we implement the use of Recursive Feature Elimination. It utilises the underlying coef_ or feature_importance attributes of the model. It ranks the features by their estimated coefficients, and removes the weakest ones once the specified number of feature is reached. Cross-validation is used to score different feature subsets and select the best scoring collection of features. 
    
    ============================================================================
    """
    
    print('I am beginning the Recursive Feature Elimination')
    
    model.fit(X,y)
    save_a_model(model,model_name='rfecv',split_no=split_no,filepath=filepath)
    #save the rfecv model
    
    scores_after_rfecv=cross_val_score(model,X,y,scoring=scoring,cv=cv)#get the estimated performance scores
    
    combination_idx_after_rfecv=transform_the_indices(model).transform(combination_idx)# check dimension of combination_idx. needs to be in 2D.
    
    save_a_npy(combination_idx_after_rfecv,npy_name='combination_idx_after_rfecv',split_no=split_no,filepath=filepath)
    #save the new combination indices after the final filter.
    
    return model, combination_idx_after_rfecv, scores_after_rfecv


def performing_sfscv(model,X,y,combination_idx,split_no,filepath,cv,scoring='r2'):
    
    """
    ___________________________________________________
    Performing mlxtend Sequential Feature Selector CV.
    ___________________________________________________
    Args:
        model (scikit-model): scikit pipe            
        X (np.asarray): Training dataset
        y (np.asarray): label target
        combination_idx (np.array): feature indices of the X.
        split_no (int):
        filepath (str):
        cv (cross-validated)
        scoring (str)
        
    Returns:
        sfscv fitted object
        sfscv_estimator: fitted sfscv estimator
        combination_idx_after_sfscv
        scores_after_sfscv
    ============================================================================
    Notes:
    
    Here, we implement the use of Sequential Feature Selector. 
    In a greedy fashion, we remove one feature at a time (forward=False), and 
    choose the subset that yields the best model with the best 
    scoring (scoring='r2') metrics in all cross-validation 
    splits. We also allow for conditional inclusion (floating=True), if any of 
    the removed feature if included back, can improve the model.
    Instead of setting a priori number of feature, we choose the subset that yileds the best score (k_features='best')
    
    ============================================================================
    
    
    """
    
    print('I am beginning the Sequential Feature Selector \n')
    
    model.fit(X,y)
    save_a_model(model,model_name='sfscv',split_no=split_no,filepath=filepath)
    #save the sfscv model
    
    scores_after_sfscv=cross_val_score(model,X,y,scoring=scoring,cv=cv)
    #get the estimated performance scores
    
    combination_idx_after_sfscv=transform_the_indices(model).transform(combination_idx)
    
    save_a_npy(combination_idx_after_sfscv,npy_name='combination_idx_after_sfscv',split_no=split_no,filepath=filepath)
    #save the new combination indices after the final filter.
    
    return model, combination_idx_after_sfscv,scores_after_sfscv

def get_the_best_model(X_test,y_test,X_trainval,y_trainval,filepath,fold_number,pipe_dict):
    """
    ___________________________
    CHECK IF FEATURE PRUNING IMPROVED THE MODEL
    This will choose the model with the best np.mean cross val score to use to 
    predict X_test,y_test and provide model_r2 score.
    ___________________________
    Args:
        X_test
        y_test
        X_trainval
        y_trainval
        filepath
        fold_number
        pipe_dict (dict) the dictionary containing the models to be compared.
    Return:
        model_r2 score
    
    """
    
    best_model_estimator=max(pipe_dict.items(),key=lambda x: x[1]['score'])[1]['estimator']
    best_model_key=max(pipe_dict.items(),key=lambda x: x[1]['score'])[0]
    best_model_estimator.fit(X_trainval,y_trainval)
    
    with open(filepath+'log.txt','a+') as file:
        file.write('use the combination after %s for split_no %d \n'%(best_model_key,fold_number))
    
    y_pred=best_model_estimator.predict(X_test)
    model_r2=r2_score(y_test,y_pred)
    
    return model_r2

def transform_the_indices(pipe,do_feature_pruning=True):
    """
    ______________________________________________________
    Transform the original indices using the fitted pipe.
    ______________________________________________________
    This is necessary because the original pipe has StandardScaler.
    Args:
        pipe (scikit pipe)
        do_feature_pruning (bool): if the pipe has rfecv or sfscv
    Return:
        feature_transform_pipe: the pipe that has the Standard Scaler removed.
    """
    x=pipe.steps[0][1].steps[:]#copy of the steps
    x.pop(1)#remove the std_scaler
    if do_feature_pruning:
        try:
            y=pipe.steps[1]#check if there is rfecv or sfscv
            x.insert(4,y)#insert at the end of the x
            feature_transform_pipe=Pipeline(x)
        except IndexError:
            feature_transform_pipe=Pipeline(x)
    else:
        feature_transform_pipe=Pipeline(x)
    return feature_transform_pipe
    
    

class myPipe(Pipeline):
    """
    Provides the underlying coef_ and feature_importances of the pipe. This is 
    required because RFECV cannot find the coef_ or feature_importances of the 
    underlying model.
    """
    def fit(self,X,y):
        super(myPipe,self).fit(X,y)
        try:
            self.coef_=self.steps[-1][-1].coef_
        except AttributeError:
            try:
                self.feature_importances_=self.steps[-1][-1].feature_importances_
            except: # in case the model is KNN
                pass
        return

class Select_Best_Correlated_features(BaseEstimator,TransformerMixin):
    """
    __________________________________________________________________________
    Calculate pearson coef b/w features and target. And choose the best number 
    of them.
    __________________________________________________________________________
    Args:
        percent= Percentage to choose from
        k_best= Number to choose from.
    """
    def __init__(self,percent=0.2,k_best=None):
        self.percent=percent
        self.k_best=k_best
        return

    def fit (self,X,y):
        self.correlation_to_the_target=np.asarray([preprocessing.lower_triangle(np.corrcoef(X[:,i],y,rowvar=True),2)[0] for i in range(X.shape[1])])
        if type(self.k_best)==int:
            self.total_features_select=self.k_best
        else:
            self.total_features_select=int(np.ceil(self.percent*X.shape[1]))
        self.features_selected=np.sort((np.argsort(self.correlation_to_the_target)[::-1][:self.total_features_select]))
        return self

    def transform(self,X,y=None):
        X_new=X[:,self.features_selected]
        return X_new

class choose_by_p_value_of_fregression(BaseEstimator,TransformerMixin):
    def __init__(self,alpha=0.05):
        self.alpha=alpha

    def fit(self,X,y):
        self.F_score,self.p_value=f_regression(X,y)
        return self
    def transform(self,X,y=None):
        X_new=X[:,np.where(self.p_value<=self.alpha)[0]]
        return X_new


class scikit_model:
    """
    Handles the fine-tuning and feature selection.
    """

    def __init__(self,model,X,y,combination_idx,fine_tune='grid',filepath='./',model_name=None,random_state=42):
        
        """
        ___________________________
        Initialize the object
        ___________________________
        Args:
            model (scikit object): the scikit model
            X (np.array): the dataset
            y (np. array): the label in 1D vector
            fine_tune (str): 'grid'= GridSearchCV, 'random'= RandomSearchCV
            filepath(str):' the file path
            model_name(str): the name of the model
            step (int or float): the name of feature removed after each step in 
                recursive feature elimination CV.
            random_state (int): random state for cross-validation 
                train_test_split.
        
        Attributes:
            self.model
            self.X
            self.y
            filepath
            self.fine_tune (str): grid vs. randomized
            self.model_name
        
        Functions:
            feature_selection_model(self,combination_idx=np.arange(4005))
        """
        
        args, _, _, values=inspect.getargvalues(inspect.currentframe())
        values.pop('self')
        
        for arg,val in values.items():
            setattr(self, arg,val)
        
        self.parameters_dict=dict({'lin_reg':None,
                         'lasso':{'alpha':[0]},
                         'ridge':{'alpha':np.linspace(200,500,10)},
                         'random_forest':None,
                         'lin_svr':{'C':[0],'epsilon':[0]},
                         'knn':{'n_neighbors':[int(i) for i in np.linspace(1,20,20)],'weights':['uniform','distance'],'metric':['minkowski','euclidean','manhattan']}})

    def feature_selection_model_simple(self,do_feature_pruning='both'):
        """
        ___________________________
        Feature selecting the model
        ___________________________
        
        The following steps are applied:
        
        First, the data is divided into 5 folds. 1 folds is saved as test set,
            and the others are for training and cross-validation.
        
        Second, for each split, a pipe is created. This pipe contains 1) 
        Variance Threshold 2) Standard Scaler 3) High Correlated Feature 
        remover 4) Univariate Test/ Select Percentile 5) Permutation Importance 
        6) Underlying model. If feature pruning is selected, then either RFECV 
        or SFSCV is applied between the 5th and 6th step of the pipe.
        
        Once the pipe is created. This is passed to a fine tuning process 
        (RandomizedCV) to identify the best features (High corr remover and 
        Select Percentile and model's associated parameters in the param_dict 
        will be fine tuned). Cross-validation is set for 4 folds. RFECV and 
        SFSCV is also set at 5 folds each. When RFECV and SFSCV is selected, 
        the model will run for 4x4 16 fits. SFSCV will take some time.
        
        
        Args:
            combination_idx= the features idx.
            do_feature_pruning (str): 'both' : do both Recursive Feature 
            elimination and SequentialFeatureSelector. Output CV score for both 
            and compare them with feature permutation importance step.
            'rfecv': Do only rfecv or 'sfs': do only sfs.
            'none' for neither.
        
        Returns:
            saves externally the combination_idx 
            self.cross_validated_scores_after_rfecv
            self.test_scores_across_all_split
        """
        inner_cv=KFold(n_splits=4,random_state=self.random_state)
        outer_cv=KFold(n_splits=5,random_state=self.random_state)
        #create inner and outer folds

        fold_number=0
        self.cross_validated_scores_after_rfecv=[]
        self.cross_validated_scores_after_sfscv=[]
        self.cross_validated_scores_after_lvr=[]
        self.test_scores_across_all_splits=[]

        for trainval_index,test_index in outer_cv.split(self.X,self.y):
            
            fold_number+=1
            print('I am starting the fold %d'%fold_number)
            X_trainval=self.X[trainval_index,:]
            y_trainval=self.y[trainval_index]
            X_test=self.X[test_index,:]
            y_test=self.y[test_index]
            combination_idx=self.combination_idx.reshape(1,-1)
            'Pipe_1: Setting initial parameters'
            
            pipe0=Pipeline([('lvr',running_model.Low_Variance_Remover(variance_percent=0)),
                           ('std_scaler',StandardScaler()),
                           ('f_reg',choose_by_p_value_of_fregression(alpha=0.05))])
            
            scaler_y=StandardScaler()
            y_trainval=scaler_y.fit_transform(y_trainval.reshape(-1,1)).ravel()
            y_test=scaler_y.transform(y_test.reshape(-1,1)).ravel()
            
            pipe_with_model=myPipe([('pipe0',pipe0),
                                    (self.model_name,self.model)])
            
            try:
                param_dict={**dict(zip([self.model_name+'__'+str(i) for i in self.parameters_dict[self.model_name].keys()],self.parameters_dict[self.model_name].values()))}
            except AttributeError:
                param_dict=AttributeError

            fine_tuned_estimator=fine_tune_hyperparameters(param_dict,pipe_with_model,X_trainval,y_trainval,model_name=self.model_name,fine_tune=self.fine_tune,cv=inner_cv,scoring='r2')
            
            scores_after_lvr=cross_val_score(fine_tuned_estimator,X_trainval,y_trainval,scoring='r2',cv=inner_cv)
            
            self.cross_validated_scores_after_lvr.append(scores_after_lvr)
            
            with open(self.filepath+'score_log.txt','a+') as file:
                file.write('cross_val_score_after_lvr (fined_tuned) for split %d is:%s'%(fold_number,','.join([str(i) for i in scores_after_lvr])))
            
            save_a_model(fine_tuned_estimator,model_name='fine_tuned_estimator',split_no=fold_number,filepath=self.filepath) #save the fine tuned esitmator
                        
            '''
            The cross-validated feature permutation importance is finished. Check if I want to do feature elimination (RFE vs. SFS or both)
            '''
            
            if do_feature_pruning=='none':
                print('Not doing feature elimination')
                
                
                fine_tuned_estimator.fit(X_trainval,y_trainval)
                combination_idx_after_lvr=transform_the_indices(fine_tuned_estimator,do_feature_pruning=False).transform(combination_idx)
                save_a_npy(combination_idx_after_lvr,npy_name='combination_idx_after_lvr',split_no=fold_number,filepath=self.filepath)
                
                y_pred=fine_tuned_estimator.predict(X_test)
                model_r2=r2_score(y_test,y_pred)
                self.test_scores_across_all_splits.append(model_r2)
                
                with open(self.filepath+'log.txt','a+') as file:
                    file.write('use the combination after perm for split_no %d \n'%fold_number)
                with open(self.filepath+'score_log.txt','a+') as file:
                    file.write('test score for split %d is: %s'%(fold_number,str(model_r2)))
                continue
            
            elif do_feature_pruning=='both':
                
                rfecv_pipe=Pipeline([('pipe0',fine_tuned_estimator.named_steps['pipe0']),
                                     ('rfecv',RFECV(fine_tuned_estimator.named_steps[self.model_name],scoring='r2',verbose=1,n_jobs=-1)),
                                     (self.model_name,fine_tuned_estimator.named_steps[self.model_name])])
                
                sfscv_pipe=Pipeline([('pipe0',fine_tuned_estimator.named_steps['pipe0']),
                                     ('sfscv',SFS(fine_tuned_estimator.named_steps[self.model_name],k_features='best',forward=False,floating=False,verbose=1,scoring='r2',n_jobs=-1)),
                                     (self.model_name,fine_tuned_estimator.named_steps[self.model_name])])
                
                rfecv_fitted_pipe,combination_idx_after_rfecv,scores_after_rfecv = performing_rfecv(rfecv_pipe,X_trainval,y_trainval,combination_idx=combination_idx,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                sfscv_fitted_pipe, combination_idx_after_sfscv,scores_after_sfscv = performing_sfscv(sfscv_pipe,X_trainval,y_trainval,combination_idx=combination_idx,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                self.cross_validated_scores_after_rfecv.append(scores_after_rfecv)
                self.cross_validated_scores_after_sfscv.append(scores_after_sfscv)
                with open(self.filepath+'score_log.txt','a+') as file:
                    file.write('cross_val_score_after_rfecv for split %d is:%s'%(fold_number,','.join([str(i) for i in scores_after_rfecv])))
                    file.write('cross_val_score_after_sfscv for split %d is:%s'%(fold_number,','.join([str(i) for i in scores_after_sfscv])))
                
                
                pipe_dict={'perm':{'estimator':fine_tuned_estimator,'score':np.mean(scores_after_lvr)},
                           'rfecv':{'estimator':rfecv_pipe,'score':np.mean(scores_after_rfecv)},
                           'sfscv':{'estimator':sfscv_pipe,'score':np.mean(scores_after_sfscv)}}
                
                model_r2= get_the_best_model(X_test,y_test,X_trainval,y_trainval,self.filepath,fold_number,pipe_dict)
                
                self.test_scores_across_all_splits.append(model_r2)
                with open(self.filepath+'score_log.txt','a+') as file:
                    file.write('test score for split %d is: %s'%(fold_number,str(model_r2)))
                continue
            
            elif do_feature_pruning=='rfecv':
                rfecv_pipe=Pipeline([('pipe0',fine_tuned_estimator.named_steps['pipe0']),
                                     ('rfecv',RFECV(fine_tuned_estimator.named_steps[self.model_name],scoring='r2',verbose=1,n_jobs=-1)),
                                     (self.model_name,fine_tuned_estimator.named_steps[self.model_name])])
                
                rfecv_fitted_pipe,combination_idx_after_rfecv,scores_after_rfecv = performing_rfecv(rfecv_pipe,X_trainval,y_trainval,combination_idx=combination_idx,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                self.cross_validated_scores_after_rfecv.append(scores_after_rfecv)
                
                with open(self.filepath+'score_log.txt','a+') as file:
                    file.write('cross_val_score_after_rfecv for split %d is:%s'%(fold_number,','.join([str(i) for i in scores_after_rfecv])))
                    
                fine_tuned_estimator.fit(X_trainval,y_trainval)
                pipe_dict={'perm':{'estimator':fine_tuned_estimator,'score':np.mean(scores_after_lvr)},
                           'rfecv':{'estimator':rfecv_pipe,'score':np.mean(scores_after_rfecv)}}
                
                model_r2= get_the_best_model(X_test,y_test,X_trainval,y_trainval,self.filepath,fold_number,pipe_dict)
                
                self.test_scores_across_all_splits.append(model_r2)
                with open(self.filepath+'score_log.txt','a+') as file:
                    file.write('test score for split %d is: %s'%(fold_number,str(model_r2)))
                continue
            
            elif do_feature_pruning=='sfscv':
                sfscv_pipe=Pipeline([('pipe0',fine_tuned_estimator.named_steps['pipe0']),
                                     ('sfscv',SFS(fine_tuned_estimator.named_steps[self.model_name],k_features='best',forward=False,floating=False,verbose=1,scoring='r2',n_jobs=-1)),
                                     (self.model_name,fine_tuned_estimator.named_steps[self.model_name])])
                
                sfscv_fitted_pipe, combination_idx_after_sfscv,scores_after_sfscv = performing_sfscv(sfscv_pipe,X_trainval,y_trainval,combination_idx=combination_idx,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                self.cross_validated_scores_after_sfscv.append(scores_after_sfscv)
                
                with open(self.filepath+'score_log.txt','a+') as file:
                    file.write('cross_val_score_after_sfscv for split %d is:%s'%(fold_number,','.join([str(i) for i in scores_after_sfscv])))
                    
                fine_tuned_estimator.fit(X_trainval,y_trainval)
                pipe_dict={'perm':{'estimator':fine_tuned_estimator,'score':np.mean(scores_after_lvr)},
                           'sfscv':{'estimator':sfscv_pipe,'score':np.mean(scores_after_sfscv)}}
                
                model_r2= get_the_best_model(X_test,y_test,X_trainval,y_trainval,self.filepath,fold_number,pipe_dict)
                
                self.test_scores_across_all_splits.append(model_r2)
                with open(self.filepath+'score_log.txt','a+') as file:
                    file.write('test score for split %d is: %s'%(fold_number,str(model_r2)))
                    
                continue
            
        return self
    
if __name__ == "__main__":
    
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import LinearSVR
    from sklearn.neighbors import KNeighborsRegressor
    
    X=np.load(sys.argv[1])
    y=np.load(sys.argv[2])
    combination_idx=np.load(sys.argv[3])
    model_dict={'lin_reg':LinearRegression(),
                'lasso':Lasso(),
                'ridge':Ridge(),
                'random_forest':RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1),
                'lin_svr':LinearSVR(),
                'knn': KNeighborsRegressor(n_jobs=-1)}
    model_name=input('model name:')
    model=model_dict[model_name]
    filepath=input('filepath:')
    # filepath='./'
    fine_tune=input('fine tune (grid/randomized):')
    x=scikit_model(model,X,y,combination_idx=combination_idx,fine_tune=fine_tune,filepath=filepath,model_name=model_name,random_state=42)
    do_feature_pruning=input('Do feature prunning?(none,both,rfecv,sfscv):')
    x.feature_selection_model_simple(do_feature_pruning=do_feature_pruning)
    
    #saving this object for logging purposes

    import pickle
    class_name=filepath+'object'+model_name+'.pkl'
    with open(class_name,'wb') as output:
        pickle.dump(x,output,pickle.HIGHEST_PROTOCOL)
    
    
    
    
    

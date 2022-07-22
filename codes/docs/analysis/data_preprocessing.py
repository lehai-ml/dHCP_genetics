""" data_preprocessing.py
This is a custom files of all preprocessing steps necessary to transform the 
diffusion and volumetric data before the data exploration and ML training step.
"""
import numpy as np
import pandas as pd
from typing import Union, List
from .data_preprocessing_high_dimension import FeatureReduction

def move_multiple_columns(df,cols_to_move=[],ref_col='',place='After'):
    """
    Move multiple columns of the pandas dataframe to a new position.
    Args:
        df (Pandas dataframe)
        cols_to_move (list): list of the name of the columns
        ref_col (str): name of the reference column
        place ('After' or 'Before'): place the new columns either after or before the reference column.
    Returns:
        a new data frame.
    
    E.g.
        move_multiple_columns(european_diffusion_dataset,cols_to_move=['Session_y','Gender','GA','PMA'],ref_col='ID',place='After')
    """
    cols=df.columns.to_list()
    if place=='After':
        seg1=cols[:cols.index(ref_col)+1]
        seg2=cols_to_move
    if place=='Before':
        seg1=cols[:cols.index(ref_col)]
        seg2=cols_to_move+[ref_col]
    seg1=[i for i in seg1 if i not in seg2]
    seg3=[i for i in cols if i not in seg1+seg2]
    return(df[seg1+seg2+seg3])

class Diffusion:
    @staticmethod
    def lower_triangle(matrix):
        """
        Works in Diffusion dataset. Organizes a square unidirectional matrix into 1D vector. Because the matrix 
            is unidirectional, only the lower triangle is needed.
        
        Args:
            matrix(np.array): the square matrix
        
        Returns:
            new_array(np.array): 1D vector of the lower half.
        Example:     
        0 1 2 3
        1 0 3 4
        2 3 0 5====>[1,2,3,3,4,5]
        3 4 5 0
        """
        m = matrix.shape[0]
        r = np.arange(m)
        mask = r[:,None] > r
        return matrix[mask]
        
    @staticmethod
    def reverse_lower_triangle(matrix,side_of_the_square=90):
        """
        Organise a 1D matrix to a square 2D undirected matrix.
        Args:
            matrix: 1D matrix
            side_of_the_square (int): desired square size 
        Returns:
            matrix: 2D matrix
        """
        def fill_in_matrix(return_matrix,matrix,side_of_the_square):
            counter=0
            for i in range(1,side_of_the_square):
                for n in range(i):
                    return_matrix[n][i]=return_matrix[i][n]=matrix[counter]
                    counter+=1
            return return_matrix
        try:
            return_matrix=np.zeros((side_of_the_square,side_of_the_square))
            return fill_in_matrix(return_matrix,matrix,side_of_the_square)
        except ValueError:
            return_matrix=np.zeros((side_of_the_square,side_of_the_square), dtype='object')
            return fill_in_matrix(return_matrix,matrix,side_of_the_square)
    
    @staticmethod
    def create_ROIs_combinations(csv_filename: str=None,
                                 ROIs=None)-> np.ndarray:
        """[Provide the combination of the Region of Interests]

        Args:
            csv_filename (str): [path to the filename in .csv]
            The file must contain 'abbr.' columns where the regions are abbreviated.

        Returns:
            np.ndarray: [list of combination in the same order as the
        #         connectivity matrices (90x90)]
        """
        if not ROIs:
            ROIs = pd.read_csv(csv_filename).dropna()
            ROIs = ROIs.loc[:,'abbr. '].str.replace(' ','')
            ROIs = ROIs.to_list()
        combinations = []
        for i in ROIs:
            combination = [i+'_'+n for n in ROIs]
            combinations.append(combination)
        combinations = np.vstack(combinations)
        return combinations
    
    @staticmethod
    def create_nodes_from_ROIs_combinations(cols):
        nodes1 = list(dict.fromkeys([i.split('_')[0] for i in cols]))
        nodes2 = list(dict.fromkeys([i.split('_')[1] for i in cols]))
        nodes = list(dict.fromkeys(nodes2+nodes1))
        return nodes
        
    
    

class Volumes:
    @staticmethod
    def Group_Imperial_volumes(df:pd.DataFrame,
                               grouping:str=None,
                               operation:str = 'sum',
                               remove_duplicated:bool=True,
                               return_grouping_scheme:bool=False)->pd.DataFrame:
        """
        Grouping the volumes of the brain regions segmented by DrawEM (Imperial atlas).
        The Imperial labels are as follows:
            white_matter_labels       = 51..82              # tissues: 3
            gray_matter_labels        = 5..16,20..39        # tissues: 2
            deep_gray_matter_labels   = 1..4,40..47,85..87  # tissues: 5,7,9
            lateral_ventricles_labels = 49,50
            corpus_callosum_labels    = 48
            inter_hemisphere_labels   = 40..47,85..87
            brainstem_labels          = 19
            cerebellum_labels         = 17,18
        Parameters
        ----------
        df : pd.DataFrame
            This is the volumetric dataframe.
        grouping: str {'segmented','gmwm2gether','all',None}
            segmented: return grouped WM and grouped GM (e.g. anterior superior temporal gyrus WM right side is grouped with posterior STG WM right side)
            gmwm2gether: return grouped WM and GM (e.g. anterior superior temporal right side WM + anterior STG right side GM)
            all: return grouped segmented and gmwm2gether - first do 'segmented' then do 'gmwm2gether'.
        operation : str, optional
            {'sum','mean'}. The default is 'sum
        remove_duplicated : bool, optional
            Remove duplicated columns after groupings. The default is True. (useful when plotting)
        return_grouping_scheme: bool, optional
            Return the grouping scheme dictionary

        Returns
        -------
        new_df : pd.DataFrame
            Grouped dataframe.

        """
        new_df = df.copy()
        
        WM_labels = [f'Imperial {i}' for i in range(51,83)] + ['Imperial 48']
        GM_labels = [f'Imperial {i}' for i in range(5,17)] + [f'Imperial {i}' for i in range(20,40)]
        DGM_labels = [f'Imperial {i}' for i in range(1,5)] + [f'Imperial {i}' for i in range(40,48)] + [f'Imperial {i}' for i in range(86,88)]
        Ventricles_labels = [f'Imperial {i}' for i in range(49,51)]
        Brainstem_labels = ['Imperial 19']
        Cerebellum_labels = ['Imperial 17','Imperial 18']
        
        Imperial_vols=new_df[[i for i in df.columns if 'Imperial' in i]].copy() # get the regions with Imperial in the name
        new_df['GM_sum_Imperial'] = Imperial_vols[GM_labels].sum(axis=1)
    
        new_df['WM_sum_Imperial'] = Imperial_vols[WM_labels].sum(axis=1)
    
        new_df['Deep_Gray_Imperial'] = Imperial_vols[DGM_labels].sum(axis=1)
    
        new_df['Ventricles_Imperial'] = Imperial_vols[Ventricles_labels].sum(axis=1)
        new_df['brainstem_Imperial'] = Imperial_vols[Brainstem_labels]
        new_df['cerebellum_Imperial'] = Imperial_vols[Cerebellum_labels].sum(axis=1)
        new_df['CSF_Imperial'] = Imperial_vols['Imperial 83']
        new_df['Intracranial_Imperial'] = new_df.loc[:,['GM_sum_Imperial','WM_sum_Imperial','Deep_Gray_Imperial','Ventricles_Imperial','brainstem_Imperial','cerebellum_Imperial','CSF_Imperial']].sum(axis=1)
        new_df['Total_Brain_Volume_Imperial'] = new_df.loc[:,['GM_sum_Imperial','WM_sum_Imperial','Deep_Gray_Imperial','brainstem_Imperial','cerebellum_Imperial']].sum(axis=1)
        if grouping is not None:
            if grouping == 'segmented':
                #Grey Matter
                new_df = FeatureReduction.combine_columns_together(new_df,[['Imperial 5','Imperial 7'],['Imperial 6','Imperial 8'], #  Anterior Temporal Lobe    
                                                                           ['Imperial 9','Imperial 25'],['Imperial 10','Imperial 24'],# Gyri parahippocampalis et ambines
                                                                           ['Imperial 11','Imperial 31'],['Imperial 12','Imperial 30'],# STG
                                                                           ['Imperial 13','Imperial 29'],['Imperial 14','Imperial 28'], # Medial and Inferior Temporal gyri
                                                                           ['Imperial 15','Imperial 27'],['Imperial 16','Imperial 26'],# Lateral occipital gyrus
                                                                           ['Imperial 33','Imperial 35'],['Imperial 32','Imperial 34']],# Cingulate gyrus
                                                          operation=operation,
                                                          remove_duplicated=remove_duplicated)
                
                #White Matter
                new_df = FeatureReduction.combine_columns_together(new_df,[['Imperial 57','Imperial 74'],['Imperial 58','Imperial 73'], # # STG
                                                                           ['Imperial 51','Imperial 53'],['Imperial 52','Imperial 54'],# this is the Anterior Temporal Lob
                                                                           ['Imperial 55','Imperial 68'],['Imperial 56','Imperial 67'],#Gyri parahippocampalis et ambines
                                                                           ['Imperial 59','Imperial 72'],['Imperial 60','Imperial 71'], # Medial and Inferior Temporal gyri
                                                                           ['Imperial 61','Imperial 70'],['Imperial 62','Imperial 69'],# Lateral occipital gyrus
                                                                           ['Imperial 76','Imperial 78'],['Imperial 75','Imperial 77']],# Cingulate gyrus
                                                          operation=operation,
                                                          remove_duplicated=remove_duplicated)
                
                #DeepGray Matter
                new_df = FeatureReduction.combine_columns_together(new_df, [['Imperial 42','Imperial 86'],
                                                                            ['Imperial 43', 'Imperial 87']],
                                                                   operation=operation,
                                                                   remove_duplicated = remove_duplicated)
            elif grouping == 'gmwm2gether':
                new_df = FeatureReduction.combine_columns_together(new_df, [['Imperial 5','Imperial 51'],['Imperial 6','Imperial 52'],['Imperial 7','Imperial 53'],['Imperial 8','Imperial 54'], #Anterior Temporal lobe
                                                                            ['Imperial 9','Imperial 55'],['Imperial 10','Imperial 56'],['Imperial 25','Imperial 68'],['Imperial 24','Imperial 67'], # Gyri parahippocampalis et ambines
                                                                            ['Imperial 11','Imperial 57'],['Imperial 12','Imperial 58'],['Imperial 31','Imperial 74'],['Imperial 30','Imperial 73'], # STG
                                                                            ['Imperial 13','Imperial 59'],['Imperial 14','Imperial 60'],['Imperial 29','Imperial 72'],['Imperial 28','Imperial 71'], # Medial and ITG
                                                                            ['Imperial 15','Imperial 61'],['Imperial 16','Imperial 62'],['Imperial 27','Imperial 70'],['Imperial 26','Imperial 69'], # Lateral Occipital Gyrus
                                                                            ['Imperial 33','Imperial 76'],['Imperial 32','Imperial 75'],['Imperial 35','Imperial 78'],['Imperial 34','Imperial 77'],# Cingulate Gyrus
                                                                            ['Imperial 21','Imperial 64'],['Imperial 20','Imperial 63'], # insula
                                                                            ['Imperial 23','Imperial 66'],['Imperial 22','Imperial 65'], # occipital lobe
                                                                            ['Imperial 37','Imperial 80'],['Imperial 36','Imperial 79'],# frontal lobe
                                                                            ['Imperial 39','Imperial 82'],['Imperial 38','Imperial 81']],# parietal lobe
                                                                   operation=operation,
                                                                   remove_duplicated = remove_duplicated)
            elif grouping == 'all':
                new_df = FeatureReduction.combine_columns_together(new_df, [['Imperial 5','Imperial 51','Imperial 7','Imperial 53'],['Imperial 6','Imperial 52','Imperial 8','Imperial 54'], #Anterior Temporal lobe
                                                                            ['Imperial 9','Imperial 55','Imperial 25','Imperial 68'],['Imperial 10','Imperial 56','Imperial 24','Imperial 67'], # Gyri parahippocampalis et ambines
                                                                            ['Imperial 11','Imperial 57','Imperial 31','Imperial 74'],['Imperial 12','Imperial 58','Imperial 30','Imperial 73'], # STG
                                                                            ['Imperial 13','Imperial 59','Imperial 29','Imperial 72'],['Imperial 14','Imperial 60','Imperial 28','Imperial 71'], # Medial and ITG
                                                                            ['Imperial 15','Imperial 61','Imperial 27','Imperial 70'],['Imperial 16','Imperial 62','Imperial 26','Imperial 69'], # Lateral Occipital Gyrus
                                                                            ['Imperial 33','Imperial 76','Imperial 35','Imperial 78'],['Imperial 32','Imperial 75','Imperial 34','Imperial 77'],# Cingulate Gyrus
                                                                            ['Imperial 21','Imperial 64','Imperial 20','Imperial 63'], # insula
                                                                            ['Imperial 23','Imperial 66'],['Imperial 22','Imperial 65'], # occipital lobe
                                                                            ['Imperial 37','Imperial 80'],['Imperial 36','Imperial 79'],# frontal lobe
                                                                            ['Imperial 39','Imperial 82'],['Imperial 38','Imperial 81'],# Parietal lobe
                                                                            ['Imperial 42','Imperial 86'],['Imperial 43','Imperial 87']], # Thalamus
                                                                            operation=operation,
                                                                            remove_duplicated = remove_duplicated)
            
        return new_df
    
    @staticmethod
    def extract_WM_Imperial(df:Union[pd.DataFrame,List])->pd.DataFrame:
        WM_labels = [f'Imperial {i}' for i in range(51,83)] + ['Imperial 48'] + ['WM_sum_Imperial']
        if isinstance(df,list):
            return [i for i in df if i in WM_labels]
        try:
            WM_df = df[df['Connection'].isin(WM_labels)].sort_values(by='PRS_pval') #search WM cols
        except KeyError: # propably not a Mass Univariate table
            WM_df = df.loc[:,df.columns.isin(WM_labels)]
        return WM_df
    
    @staticmethod
    def extract_GM_Imperial(df:Union[pd.DataFrame,List])->pd.DataFrame:
        GM_labels = [f'Imperial {i}' for i in range(5,17)] + [f'Imperial {i}' for i in range(20,40)] + ['GM_sum_Imperial']
        if isinstance(df,list):
            return [i for i in df if i in GM_labels]
        try:
            GM_df = df[df['Connection'].isin(GM_labels)].sort_values(by='PRS_pval') #search WM cols
        except KeyError: # propably not a Mass Univariate table
            GM_df = df.loc[:,df.columns.isin(GM_labels)]
        return GM_df
    
    @staticmethod
    def extract_deepGM_Imperial(df:Union[pd.DataFrame,List])->pd.DataFrame:
        DGM_labels = [f'Imperial {i}' for i in range(1,5)] + [f'Imperial {i}' for i in range(40,48)] + [f'Imperial {i}' for i in range(86,88)]
        if isinstance(df,list):
            return [i for i in df if i in DGM_labels]
        try:
            DGM_df = df[df['Connection'].isin(DGM_labels)].sort_values(by='PRS_pval') #search WM cols
        except KeyError: # propably not a Mass Univariate table
            DGM_df = df.loc[:,df.columns.isin(DGM_labels)]
        return DGM_df
    
    @staticmethod
    def Group_AAL_volumes(df:pd.DataFrame,
                        grouping:bool=True,
                        operation:str = 'sum',
                        remove_duplicated:bool=True)->pd.DataFrame:
        """
        Grouping the volumes of the brain regions segmented by AAL atlas.
        The following AAL volumes are grouped
        Parameters
        ----------
        df : pd.DataFrame
            This is the volumetric dataframe.
        operation : str, optional
            {'sum','mean'}. The default is 'sum
        remove_duplicated : bool, optional
            Remove duplicated columns after groupings. The default is True.

        Returns
        -------
        new_df : pd.DataFrame
            Grouped dataframe.

        """
        new_df = df.copy()
        if grouping:
            new_df = FeatureReduction.combine_columns_together(new_df,
                                                               group_columns=[['AAL 3','AAL 23'],['AAL 4','AAL 24'],#Superior frontal gyrus
                                                                              ['AAL 5','AAL 9','AAL 15','AAL 25'],['AAL 6','AAL 10','AAL 16','AAL 26'], # Orbitofrontal cortex
                                                                              ['AAL 11','AAL 13'],['AAL 12','AAL 14'], # Inferior frontal gyrus
                                                                              ['AAL 83','AAL 87'],['AAL 84','AAL 88']],operation = operation,remove_duplicated = remove_duplicated) # Temporal pole
        return new_df
    
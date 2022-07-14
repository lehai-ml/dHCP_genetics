""" data_preprocessing.py
This is a custom files of all preprocessing steps necessary to transform the 
diffusion and volumetric data before the data exploration and ML training step.
"""
import numpy as np
import pandas as pd

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
    def Group_Imperial_volumes(df:pd.DataFrame,grouping:bool=True,remove_duplicated:bool=True)->pd.DataFrame:
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
        remove_duplicated : bool, optional
            Remove duplicated columns after groupings. The default is True.

        Returns
        -------
        new_df : pd.DataFrame
            Grouped dataframe.

        """
        new_df = df.copy()
        
        WM_labels = [f'Imperial {i}' for i in range(51,83)] + ['Imperial 48']
        GM_labels = [f'Imperial {i}' for i in range(5,17)] + [f'Imperial {i}' for i in range(20,40)]
        DGM_labels = [f'Imperial {i}' for i in range(1,5)] + [f'Imperial {i}' for i in range(40,48)] + [f'Imperial {i}' for i in range(85,88)]
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
        
        if grouping:
            def grouping(new_df,left=None,right=None,remove_duplicated=True):
                #group the anterior to the posterior part
                new_df[left[0]] = new_df[left[1]] = new_df.loc[:,left].sum(axis = 1) # sum the left regions up
                new_df[right[0]] = new_df[right[1]] = new_df.loc[:,right].sum(axis = 1) # sum the right regions up
                if remove_duplicated:
                    new_df.drop(columns = [left[1],right[1]],inplace=True) # drop the duplicated regions and retain only one
        
            #Grey Matter
            grouping(new_df,['Imperial 5','Imperial 7'],['Imperial 6','Imperial 8'],remove_duplicated) #  Anterior Temporal Lobe    
            grouping(new_df,['Imperial 9','Imperial 25'],['Imperial 10','Imperial 24'],remove_duplicated) # Gyri parahippocampalis et ambines
            grouping(new_df,['Imperial 11','Imperial 31'],['Imperial 12','Imperial 30'],remove_duplicated) # STG
            grouping(new_df,['Imperial 13','Imperial 29'],['Imperial 14','Imperial 28'],remove_duplicated) # Medial and Inferior Temporal gyri
            grouping(new_df,['Imperial 15','Imperial 27'],['Imperial 16','Imperial 26'],remove_duplicated) # Lateral occipital gyrus
            grouping(new_df,['Imperial 33','Imperial 35'],['Imperial 32','Imperial 34'],remove_duplicated) # Cingulate gyrus
                
            #White Matter
            grouping(new_df,['Imperial 57','Imperial 74'],['Imperial 58','Imperial 73'],remove_duplicated) # STG
            grouping(new_df,['Imperial 51','Imperial 53'],['Imperial 52','Imperial 54'],remove_duplicated) # this is the Anterior Temporal Lob
            grouping(new_df,['Imperial 55','Imperial 68'],['Imperial 56','Imperial 67'],remove_duplicated) # Gyri parahippocampalis et ambines
            grouping(new_df,['Imperial 59','Imperial 72'],['Imperial 60','Imperial 71'],remove_duplicated) # Medial and Inferior Temporal gyri
            grouping(new_df,['Imperial 61','Imperial 70'],['Imperial 62','Imperial 69'],remove_duplicated) # Lateral occipital gyrus
            grouping(new_df,['Imperial 75','Imperial 77'],['Imperial 76','Imperial 78'],remove_duplicated) # Cingulate gyrus
        
        return new_df
    
    @staticmethod
    def extract_WM_Imperial(df:pd.DataFrame)->pd.DataFrame:
        WM_labels = [f'Imperial {i}' for i in range(51,83)] + ['Imperial 48'] + ['WM_sum_Imperial']
        try:
            WM_df = df[df['Connection'].isin(WM_labels)].sort_values(by='PRS_pval') #search WM cols
        except KeyError: # propably not a Mass Univariate table
            WM_df = df.loc[:,df.columns.isin(WM_labels)]
        return WM_df
    
    @staticmethod
    def extract_GM_Imperial(df:pd.DataFrame)->pd.DataFrame:
        GM_labels = [f'Imperial {i}' for i in range(5,17)] + [f'Imperial {i}' for i in range(20,40)] + ['GM_sum_Imperial']
        try:
            GM_df = df[df['Connection'].isin(GM_labels)].sort_values(by='PRS_pval') #search WM cols
        except KeyError: # propably not a Mass Univariate table
            GM_df = df.loc[:,df.columns.isin(GM_labels)]
        return GM_df
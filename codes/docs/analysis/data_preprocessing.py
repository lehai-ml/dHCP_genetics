""" data_preprocessing.py
This is a custom files of all preprocessing steps necessary to transform the 
diffusion and volumetric data before the data exploration and ML training step.

Following steps are available:
    
    move_multiple_columns - used to move multiple columns of the pandas dataframe to a new pos.
    Diffusion
        lower_triangle- used to get the lower triangle of a symmetrical matrix
        reverse_lower_triangle- organise a 1D matrix to a square 2D undirected matrix
        create_ROIs_combinations- Provide combination of the ROI
        create_nodes_from_ROI_combinations
    Volumes
        Imperial
            group_Imperial_Volumes - Group volumes segmented by DrawEM atlas
            get_segment- Get specific segment from DrawEM legend
            get_Imperial_legends- Hardcoded Imperial legends
        AAL
            group_AAL_volumes - Group volumes segmented by AAL atlas
            
"""
import numpy as np
import pandas as pd
from typing import Union, List
from collections import defaultdict
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
        
    class WM_parcellation:
        @staticmethod
        def get_segment(df:Union[pd.DataFrame,List,dict],col:str='column',
                        **kwargs):
            if 'side' not in kwargs:
                kwargs['side'] = []
            if isinstance(kwargs['side'],str):
                kwargs['side'] = [kwargs['side']]
            def return_segment(WM_parc,tissues:list=None):
                left = [f'wm_{i}' for i in range(94,137,2)]+[f'wm_{i}' for i in range(139,142,2)]
                right = [f'wm_{i}' for i in range(95,138,2)]+[f'wm_{i}' for i in range(140,143,2)]
                all_tissues = []
                for tissue in tissues:
                    if tissue == 'left':
                        all_tissues += [i for i in WM_parc if i in left]
                    elif tissue == 'right':
                        all_tissues += [i for i in WM_parc if i in right]
                return list(set(all_tissues))
            
            if isinstance(df,list):
                segments_to_examine = df
            elif isinstance(df,pd.DataFrame):
                if isinstance(col,str):
                    if col == 'index':
                        segments_to_examine = df.index.tolist()
                    elif col == 'column':
                        segments_to_examine = df.columns.tolist()
                    else:
                        segments_to_examine = df[col].tolist()
                else:
                    raise ValueError('col must be defined if parsing dataframe')
            elif isinstance(df,dict):
                segments_to_examine = [i for i in df.keys()]
            
            if len(kwargs['side'])>=1:
                segments_to_examine = return_segment(segments_to_examine,tissues=kwargs['side'])
            
            segments_to_examine = sorted(segments_to_examine)
            if isinstance(df,list):
                return segments_to_examine
            elif isinstance(df,pd.DataFrame):
                if isinstance(col,str):
                    if col == 'index':
                        return df.loc[segments_to_examine,:]
                    elif col == 'column':
                        return df[segments_to_examine]
                    else:
                        return df.loc[df[col].isin(segments_to_examine),:]
            elif isinstance(df,dict):
                return {k:df[k] for k in segments_to_examine}
                    
        @staticmethod
        def get_wm_parcellation(grouping:dict=None,
                                df:[pd.DataFrame,list]=None,**kwargs)->dict:
            """
            Generate WM parcellation legends. May be used for visualisation.Brainmap

            Parameters
            ----------
            grouping : {'segmented','hemisphere'}, optional
                segmented - combine the same region - anterior and posterior
                hemisphere - combine the same region - left and right. The default is None.
            Returns
            -------
            label_dict : dict
                Dictionary containing the regions name.

            """
            ###This bit is hardcoded
            labels =  {
                'wm_94': 'anterior limb of internal capsule left', 
                'wm_95': 'anterior limb of internal capsule right', 
                'wm_96': 'posterior limb of internal capsule left',
                'wm_97': 'posterior limb of internal capsule right',
                'wm_98': 'retrolenticular part of internal capsule left',
                'wm_99': 'retrolenticular part of internal capsule right',
                'wm_100': 'anterior corona radiata left',
                'wm_101': 'anterior corona radiata right',
                'wm_102': 'superior corona radiata left',
                'wm_103': 'superior corona radiata right',
                'wm_104': 'posterior corona radiata left',
                'wm_105': 'posterior corona radiata right',
                'wm_106': 'cingulum cingular part left',
                'wm_107': 'cingulum cingular part right',
                'wm_108': 'cingulum hipocampal part left',
                'wm_109': 'cingulum hipocampal part right',
                'wm_110': 'fornix left',
                'wm_111': 'fornix right',
                'wm_112': 'stria terminalis left',
                'wm_113': 'stria terminalis right',
                'wm_114': 'tapetum left',
                'wm_115': 'tapetum right',
                'wm_116': 'superior longitudinal fasciculus left',
                'wm_117': 'superior longitudinal fasciculus right',
                'wm_118': 'external capsule left',
                'wm_119': 'external capsule right',
                'wm_120': 'posterior thalamic radiation left',
                'wm_121': 'posterior thalamic radiation right',
                'wm_122': 'sagittal stratum left',
                'wm_123': 'sagittal stratum right',
                'wm_124': 'cerebral peduncle left',
                'wm_125': 'cerebral peduncle right',
                'wm_126': 'superior fronto occipital fasciculus left',
                'wm_127': 'superior fronto occipital fasciculus right',
                'wm_128': 'inferior fronto occipital fasciculus left',
                'wm_129': 'inferior fronto occipital fasciculus right',
                'wm_130': 'corticospinal tract left',
                'wm_131': 'corticospinal tract right',
                'wm_132': 'superior cerebellar peduncle left',
                'wm_133': 'superior cerebellar peduncle right',
                'wm_134': 'middle cerebellar peduncle left',
                'wm_135': 'middle cerebellar peduncle right',
                'wm_136': 'inferior cerebellar peduncle left',
                'wm_137': 'inferior cerebellar peduncle right',
                'wm_138': 'pontine crossing',
                'wm_139': 'uncinate fasciculus left',
                'wm_140': 'uncinate fasciculus right',
                'wm_141': 'medial lemniscus left',
                'wm_142': 'medial lemniscus right',
                'wm_143': 'corpus callosum 1', #prefrontal part',
                'wm_144': 'corpus callosum 2', # premotor/supplementary part',
                'wm_145': 'corpus callosum 3', #motor part',
                'wm_146': 'corpus callosum 4', #sensory part',
                'wm_147': 'corpus callosum 5' #parietal/temporal/occipital part'
            }
             
            label_dict = {k:{'name':v,'side':None,
                             'segment':None,'structure':None,
                             'abbr':None} for k,v in labels.items()}
            # get basic structure of the dictionary, for each label
            # you have 1) tissue 2) lobe 3) side
            for side in ['left','right']:
                side_key = Diffusion.WM_parcellation.get_segment(label_dict,side=[side]).keys()
                for k in side_key:
                    label_dict[k]['side'] = side

            for k in label_dict:
                if 'part' in label_dict[k]['name']:
                    label_dict[k]['segment']=label_dict[k]['name'].split('part')[0].split(' ')[-2]
                elif 'limb' in label_dict[k]['name']:
                    label_dict[k]['segment']=label_dict[k]['name'].split('limb')[0].split(' ')[-2]
                
            for k in label_dict:
                label_dict[k]['structure'] = label_dict[k]['name']
                for n in label_dict[k]:
                    if n == 'name' or n=='structure':
                        continue
                    try:
                        label_dict[k]['structure'] = label_dict[k]['structure'].replace(label_dict[k][n],'')
                    except TypeError:
                        continue
                label_dict[k]['structure'] = label_dict[k]['structure'].split(',')[0]
                label_dict[k]['structure'] = label_dict[k]['structure'].replace('part','').replace('limb','').replace('of','').strip()
            
            # the abbr will be structure.segment.side
            # if tissue:
            for k in label_dict:
                if len(label_dict[k]['structure'].split()) > 1:
                    label_dict[k]['abbr'] = ''.join([x[0].upper() for x in label_dict[k]['structure'].split()])
                else:
                    label_dict[k]['abbr'] = label_dict[k]['structure'][0:4].upper()
                for n in ['side','segment']:
                        try:
                            label_dict[k]['abbr'] = '.'.join([label_dict[k]['abbr'],label_dict[k][n]])
                        except TypeError:
                            continue

            if isinstance(grouping,str):
                grouping = [grouping]
            if isinstance(grouping,(list,tuple)):
                for k in label_dict:
                    if label_dict[k]['abbr'] is not None:
                        if 'segmented' in grouping and 'lobe' not in grouping:
                            if label_dict[k]['segment'] is not None:
                                label_dict[k]['abbr'] = label_dict[k]['abbr'].replace('.'+label_dict[k]['segment'],'')
                        if 'hemisphere' in grouping:
                            if label_dict[k]['side'] is not None:
                                label_dict[k]['abbr'] = label_dict[k]['abbr'].replace('.'+label_dict[k]['side'],'')
            if df is not None:
                #if dataframe is passed, it will try to change the names of where the columns with wm parcellation are
                if isinstance(df,pd.DataFrame):
                    col = kwargs.get('col','index')
                    if col == 'index' or col == 'columns':
                        return df.rename({k:v['abbr'] for k,v in label_dict.items()},axis=col)
                    else:
                        df[col] = df[col].map({k:v['abbr'] for k,v in label_dict.items()})
                        return df
                elif isinstance(df,list):
                    return [label_dict[k]['abbr'] for k in df]
                else:
                    raise TypeError('df must be dataframe or list of strings')
            else:
                return label_dict
        
        def group_wm_parcellation(df:pd.DataFrame,grouping:[list,tuple,str]=None,
                                   operation:str='sum',drop_duplicates:bool=True):
            """
            Group Imperial Volume by summing them.
            Use grouping scheme as defined in get_wm_parcellation()
            
            NOTE: even if the df doesn't contain the necessary regions
            i.e., df was applied get_segments before grouped.
            it will still work.
            
            Parameters
            ----------
            df : pd.DataFrame
                The data Frame containing the Imperial volumes as columns.
            grouping : {'segmented','hemisphere'}, optional
                segmented - combine the same region - anterior and posterior
                hemisphere - combine the same region - left and right. The default is None.
            operation: str. {'sum','mean'}
                how to group them. (mean can be used when generating positional info)
            drop_duplicates: bool
                Whether to drop the columns after grouping process.
                by default, the first column in the group list is retained as the total
                value of the rest of the values in the list and all values but the first one
                is removed.
                If False. All values in the grouped list will have the same values.
                This is needed when plotting the Brain Map. Otherwise only the first value will show.
            Returns
            -------
            new_df : pd.dataFrame
                grouped DataFrame.

            """
            if not isinstance(df,pd.DataFrame):
                raise TypeError('Needs pandas DataFrame')
            new_df = df.copy()
            grouped_volumes_dict = Diffusion.WM_parcellation.get_wm_parcellation(grouping)
            grouped_volumes_dict = {k:v['abbr'] for k,v in grouped_volumes_dict.items() if k in new_df.columns}#{'wm_94:'IC.left.anterior'}
            unique_grouped_volumes_dict = defaultdict(list)
            for k,v in grouped_volumes_dict.items():
                unique_grouped_volumes_dict[v].append(k)
            for k,v in unique_grouped_volumes_dict.items():
                if k is None:
                    new_df = new_df.drop(columns = v)
                else:
                    if len(v) > 1:
                        if operation == 'sum':
                            new_df[v[0]] = df[v].sum(axis=1)
                        elif operation == 'mean':
                            new_df[v[0]] = df[v].mean(axis=1)
                        if drop_duplicates:
                            new_df = new_df.drop(columns=[i for i in df.columns if i in v[1:]])
                        else:
                            for col in v[1:]:
                                if col in df.columns:
                                    new_df[col] = new_df[v[0]]
            return new_df
    
class Volumes:
    class Imperial:
        
        @staticmethod
        def get_segment(df:Union[pd.DataFrame,List,dict],col:str='column',**kwargs):
            """
            Get the specific segment from DrawEM legend
            
            NOTE:
                The Imperial labels are as follows:
                    white_matter_labels       = 51..82+48              # tissues: 3
                    gray_matter_labels        = 5..16,20..39        # tissues: 2
                    deep_gray_matter_labels   = 1..4,40..47,85..87  # tissues: 5,7,9
                    lateral_ventricles_labels = 49,50
                    corpus_callosum_labels    = 48
                    inter_hemisphere_labels   = 40..47,85..87
                    brainstem_labels          = 19
                    cerebellum_labels         = 17,18
            Parameters
            ----------
            df : Union[pd.DataFrame,List,dict]
                List of labels to select from must be named ['Imperial 1', 'Imperial 2'] etc.
                can be dataframe, list or dict
            col : str, optional
                if providing dataframe, must specify where to look for labels
                {'index','column' or name of the column}. The default is None.
            **kwargs : dict
                {'tissue':[WM,GM,DGM,Ventricle,Brainstem,Cerebellum,Background,'CSF],
                 'lobe':[frontal,parietal,occipital,temporal],
                 'side':[left,right]}
            """
            if 'tissue' not in kwargs:
                kwargs['tissue'] = []
            if 'lobe' not in kwargs:
                kwargs['lobe'] = []
            if 'side' not in kwargs:
                kwargs['side'] = []
            if isinstance(kwargs['tissue'],str):
                kwargs['tissue'] = [kwargs['tissue']]
            if isinstance(kwargs['lobe'],str):
                kwargs['lobe'] = [kwargs['lobe']]
            if isinstance(kwargs['side'],str):
                kwargs['side'] = [kwargs['side']]
            
            def return_segment(Imperial_tissue:list,tissues:list=None):
                ### tissue types
                WM_labels = [f'Imperial {i}' for i in range(51,83)] + ['Imperial 48']
                GM_labels = [f'Imperial {i}' for i in range(5,17)] + [f'Imperial {i}' for i in range(20,40)]
                DGM_labels = [f'Imperial {i}' for i in range(1,5)] + [f'Imperial {i}' for i in range(40,48)] + [f'Imperial {i}' for i in range(85,88)]
                #this deep Gray matter contains the hippocampus + amygdala (in the 9TT segmentation file, deepGM do not contain Hipp+Amygdala)
                Ventricles_labels = [f'Imperial {i}' for i in range(49,51)]
                Brainstem_labels = ['Imperial 19']
                Cerebellum_labels = ['Imperial 17','Imperial 18']
                Background_labels = ['Imperial 84','Imperial 85']
                CSF_labels = ['Imperial 83']
                ### lobes
                temporal_lobes = [f'Imperial {i}' for i in [5,6,7,8,11,12,13,14,28,29,30,31,
                                                           51,52,53,54,57,58,59,60,71,72,73,74]]
                frontal_lobes = [f'Imperial {i}' for i in [36,37,79,80]]
                occipital_lobes = [f'Imperial {i}' for i in [15,16,22,23,65,66,61,62,69,70,26,27]]
                parietal_lobes = [f'Imperial {i}' for i in [38,39,81,82]]

                ###side
                left = [i for i in range(1,18,2)] + [i for i in range(21,48,2)] + [i for i in range(49,62,2)] + [i for i in range(64,83,2)] + [87]
                right = [i for i in range(1,88) if i not in left + [19,48,83,84,85]]
                left = [f'Imperial {i}' for i in left]
                right = [f'Imperial {i}' for i in right]
                
               
                all_tissues = []
                for tissue in tissues:
                    if tissue == 'WM':
                        all_tissues += [i for i in Imperial_tissue if i in WM_labels]
                    elif tissue == 'GM':
                        all_tissues += [i for i in Imperial_tissue if i in GM_labels]
                    elif tissue == 'DGM':
                        all_tissues += [i for i in Imperial_tissue if i in DGM_labels]
                    elif tissue=='Ventricle':
                        all_tissues += [i for i in Imperial_tissue if i in Ventricles_labels]
                    elif tissue == 'Brainstem':
                        all_tissues += [i for i in Imperial_tissue if i in Brainstem_labels]
                    elif tissue == 'Cerebellum':
                        all_tissues += [i for i in Imperial_tissue if i in Cerebellum_labels]
                    elif tissue == 'CSF':
                        all_tissues += [i for i in Imperial_tissue if i in CSF_labels]
                    elif tissue == 'Background':
                        all_tissues += [i for i in Imperial_tissue if i in Background_labels]
                    elif tissue=='temporal':
                        all_tissues += [i for i in Imperial_tissue if i in temporal_lobes]
                    elif tissue=='frontal':
                        all_tissues += [i for i in Imperial_tissue if i in frontal_lobes]
                    elif tissue=='occipital':
                        all_tissues += [i for i in Imperial_tissue if i in occipital_lobes]
                    elif tissue=='parietal':
                        all_tissues += [i for i in Imperial_tissue if i in parietal_lobes]
                    elif tissue == 'left':
                        all_tissues += [i for i in Imperial_tissue if i in left]
                    elif tissue == 'right':
                        all_tissues += [i for i in Imperial_tissue if i in right]

                return list(set(all_tissues))
            
            if isinstance(df,list):
                segments_to_examine = df
            elif isinstance(df,pd.DataFrame):
                if isinstance(col,str):
                    if col == 'index':
                        segments_to_examine = df.index.tolist()
                    elif col == 'column':
                        segments_to_examine = df.columns.tolist()
                    else:
                        segments_to_examine = df[col].tolist()
                else:
                    raise ValueError('col must be defined if parsing dataframe')
            elif isinstance(df,dict):
                segments_to_examine = [i for i in df.keys()]
            
            if len(kwargs['tissue'])>=1:
                segments_to_examine = return_segment(segments_to_examine,tissues=kwargs['tissue'])
            if len(kwargs['lobe'])>=1:
                segments_to_examine = return_segment(segments_to_examine,tissues=kwargs['lobe'])
            if len(kwargs['side'])>=1:
                segments_to_examine = return_segment(segments_to_examine,tissues=kwargs['side'])
            
            segments_to_examine = sorted(segments_to_examine)
            if isinstance(df,list):
                return segments_to_examine
            elif isinstance(df,pd.DataFrame):
                if isinstance(col,str):
                    if col == 'index':
                        return df.loc[segments_to_examine,:]
                    elif col == 'column':
                        return df[segments_to_examine]
                    else:
                        return df.loc[df[col].isin(segments_to_examine),:]
            elif isinstance(df,dict):
                return {k:df[k] for k in segments_to_examine}

        
                
        
        @staticmethod
        def get_Imperial_legends(grouping:[list,tuple,str]=None,
                                 df:[pd.DataFrame,list]=None,**kwargs)->dict:
            """
            Generate Imperial legends. May be used for visualisation.Brainmap

            Parameters
            ----------
            grouping : {'lobe','gmwm2gether','segmented','hemisphere'}, optional
                lobe- combine the regions by lobes (frontal, temporal, occipital, parietal)
                gmwm2gether- combine the same region- white and grey matter.
                segmented - combine the same region - anterior and posterior
                hemisphere - combine the same region - left and right. The default is None.

            Returns
            -------
            label_dict : dict
                Dictionary containing the regions name.

            """
            ###This bit is hardcoded
            labels =  {'Imperial 1': 'Hippocampus left', 
             'Imperial 2': 'Hippocampus right', 
             'Imperial 3': 'Amygdala left', 
             'Imperial 4': 'Amygdala right', 
             'Imperial 5': 'Anterior temporal lobe, medial part left GM', 
             'Imperial 6': 'Anterior temporal lobe, medial part right GM', 
             'Imperial 7': 'Anterior temporal lobe, lateral part left GM', 
             'Imperial 8': 'Anterior temporal lobe, lateral part right GM', 
             'Imperial 9': 'Gyri parahippocampalis et ambiens anterior part left GM', 
             'Imperial 10': 'Gyri parahippocampalis et ambiens anterior part right GM', 
             'Imperial 11': 'Superior temporal gyrus, middle part left GM', 
             'Imperial 12': 'Superior temporal gyrus, middle part right GM', 
             'Imperial 13': 'Medial and inferior temporal gyri anterior part left GM', 
             'Imperial 14': 'Medial and inferior temporal gyri anterior part right GM', 
             'Imperial 15': 'Lateral occipitotemporal gyrus, gyrus fusiformis anterior part left GM', 
             'Imperial 16': 'Lateral occipitotemporal gyrus, gyrus fusiformis anterior part right GM', 
             'Imperial 17': 'Cerebellum left', 
             'Imperial 18': 'Cerebellum right', 
             'Imperial 19': 'Brainstem, spans the midline', 
             'Imperial 20': 'Insula right GM', 
             'Imperial 21': 'Insula left GM', 
             'Imperial 22': 'Occipital lobe right GM', 
             'Imperial 23': 'Occipital lobe left GM', 
             'Imperial 24': 'Gyri parahippocampalis et ambiens posterior part right GM', 
             'Imperial 25': 'Gyri parahippocampalis et ambiens posterior part left GM', 
             'Imperial 26': 'Lateral occipitotemporal gyrus, gyrus fusiformis posterior part right GM', 
             'Imperial 27': 'Lateral occipitotemporal gyrus, gyrus fusiformis posterior part left GM', 
             'Imperial 28': 'Medial and inferior temporal gyri posterior part right GM', 
             'Imperial 29': 'Medial and inferior temporal gyri posterior part left GM', 
             'Imperial 30': 'Superior temporal gyrus, posterior part right GM', 
             'Imperial 31': 'Superior temporal gyrus, posterior part left GM', 
             'Imperial 32': 'Cingulate gyrus, anterior part right GM', 
             'Imperial 33': 'Cingulate gyrus, anterior part left GM', 
             'Imperial 34': 'Cingulate gyrus, posterior part right GM', 
             'Imperial 35': 'Cingulate gyrus, posterior part left GM', 
             'Imperial 36': 'Frontal lobe right GM', 
             'Imperial 37': 'Frontal lobe left GM', 
             'Imperial 38': 'Parietal lobe right GM', 
             'Imperial 39': 'Parietal lobe left GM', 
             'Imperial 40': 'Caudate nucleus right', 
             'Imperial 41': 'Caudate nucleus left', 
             'Imperial 42': 'Thalamus right, high intensity part in T2', 
             'Imperial 43': 'Thalamus left, high intensity part in T2', 
             'Imperial 44': 'Subthalamic nucleus right', 
             'Imperial 45': 'Subthalamic nucleus left', 
             'Imperial 46': 'Lentiform Nucleus right', 
             'Imperial 47': 'Lentiform Nucleus left', 
             'Imperial 48': 'Corpus Callosum', 
             'Imperial 49': 'Lateral Ventricle left', 
             'Imperial 50': 'Lateral Ventricle right', 
             'Imperial 51': 'Anterior temporal lobe, medial part left WM', 
             'Imperial 52': 'Anterior temporal lobe, medial part right WM', 
             'Imperial 53': 'Anterior temporal lobe, lateral part left WM', 
             'Imperial 54': 'Anterior temporal lobe, lateral part right WM', 
             'Imperial 55': 'Gyri parahippocampalis et ambiens anterior part left WM', 
             'Imperial 56': 'Gyri parahippocampalis et ambiens anterior part right WM', 
             'Imperial 57': 'Superior temporal gyrus, middle part left WM', 
             'Imperial 58': 'Superior temporal gyrus, middle part right WM', 
             'Imperial 59': 'Medial and inferior temporal gyri anterior part left WM', 
             'Imperial 60': 'Medial and inferior temporal gyri anterior part right WM', 
             'Imperial 61': 'Lateral occipitotemporal gyrus, gyrus fusiformis anterior part left WM', 
             'Imperial 62': 'Lateral occipitotemporal gyrus, gyrus fusiformis anterior part right WM',
             'Imperial 63': 'Insula right WM', 
             'Imperial 64': 'Insula left WM', 
             'Imperial 65': 'Occipital lobe right WM', 
             'Imperial 66': 'Occipital lobe left WM', 
             'Imperial 67': 'Gyri parahippocampalis et ambiens posterior part right WM', 
             'Imperial 68': 'Gyri parahippocampalis et ambiens posterior part left WM', 
             'Imperial 69': 'Lateral occipitotemporal gyrus, gyrus fusiformis posterior part right WM',
             'Imperial 70': 'Lateral occipitotemporal gyrus, gyrus fusiformis posterior part left WM', 
             'Imperial 71': 'Medial and inferior temporal gyri posterior part right WM', 
             'Imperial 72': 'Medial and inferior temporal gyri posterior part left WM', 
             'Imperial 73': 'Superior temporal gyrus, posterior part right WM', 
             'Imperial 74': 'Superior temporal gyrus, posterior part left WM', 
             'Imperial 75': 'Cingulate gyrus, anterior part right WM', 
             'Imperial 76': 'Cingulate gyrus, anterior part left WM', 
             'Imperial 77': 'Cingulate gyrus, posterior part right WM', 
             'Imperial 78': 'Cingulate gyrus, posterior part left WM', 
             'Imperial 79': 'Frontal lobe right WM', 
             'Imperial 80': 'Frontal lobe left WM', 
             'Imperial 81': 'Parietal lobe right WM', 
             'Imperial 82': 'Parietal lobe left WM', 
             'Imperial 83': 'CSF', 
             'Imperial 84': 'Extra-cranial background', 
             'Imperial 85': 'Intra-cranial background', 
             'Imperial 86': 'Thalamus right, low intensity part in T2', 
             'Imperial 87': 'Thalamus left, low intensity part in T2'}
            
            label_dict = {k:{'name':v,'tissue':None,
                             'lobe':None,'side':None,
                             'segment':None,'structure':None,
                             'abbr':None} for k,v in labels.items()}
            # get basic structure of the dictionary, for each label
            # you have 1) tissue 2) lobe 3) side
            for tissue in ['WM','GM','Ventricle','Brainstem','Cerebellum','Background','CSF']:
                tissue_key = Volumes.Imperial.get_segment(label_dict,tissue=[tissue]).keys()
                for k in tissue_key:
                    label_dict[k]['tissue'] = tissue
            
            for lobe in ['frontal','temporal','occipital','parietal']:
                lobe_key = Volumes.Imperial.get_segment(label_dict,lobe=[lobe]).keys()
                for k in lobe_key:
                    label_dict[k]['lobe'] = lobe
            
            for side in ['left','right']:
                side_key = Volumes.Imperial.get_segment(label_dict,side=[side]).keys()
                for k in side_key:
                    label_dict[k]['side'] = side
                    
            # for each part you check if it is medial, anterior posterior etc.            
            for k in label_dict:
                if 'part' in label_dict[k]['name']:
                    if 'Thalamus' in label_dict[k]['name']:
                        label_dict[k]['segment'] = ' '.join(label_dict[k]['name'].split('part')[0].split(' ')[-3:-1])
                    else:
                        label_dict[k]['segment']=label_dict[k]['name'].split('part')[0].split(' ')[-2]
            # finally you assign structure name to it
            for k in label_dict:
                label_dict[k]['structure'] = label_dict[k]['name']
                for n in label_dict[k]:
                    if n == 'name' or n=='structure' or n == 'lobe':
                        continue
                    if n == 'tissue' and (label_dict[k]['tissue'] in ['Cerebellum','CSF','Brainstem','Ventricle']):
                        continue
                    try:
                        label_dict[k]['structure'] = label_dict[k]['structure'].replace(label_dict[k][n],'')
                    except TypeError:
                        continue
                label_dict[k]['structure'] = label_dict[k]['structure'].split(',')[0]
                label_dict[k]['structure'] = label_dict[k]['structure'].replace('part','').strip()
            
            ### This bit is not
        
            # the abbr will be structure.tissue.segment.side
            # if tissue:
            for k in label_dict:
                if len(label_dict[k]['structure'].split()) > 1:
                    label_dict[k]['abbr'] = ''.join([x[0].upper() for x in label_dict[k]['structure'].split()])
                else:
                    label_dict[k]['abbr'] = label_dict[k]['structure'][0:4].upper()
                for n in ['tissue','side','segment']:
                        try:
                            label_dict[k]['abbr'] = '.'.join([label_dict[k]['abbr'],label_dict[k][n]])
                        except TypeError:
                            continue

            if isinstance(grouping,str):
                grouping = [grouping]
            if isinstance(grouping,(list,tuple)):#{lobe}
                if 'lobe' in grouping: # replace the structure name with lobe
                    for k in label_dict:
                        if label_dict[k]['lobe'] is None:
                            label_dict[k]['abbr'] = None
                        else:
                            label_dict[k]['abbr'] = label_dict[k]['abbr'].replace(label_dict[k]['abbr'].split('.')[0],label_dict[k]['lobe'])
                            if label_dict[k]['segment'] is not None:
                                label_dict[k]['abbr'] = label_dict[k]['abbr'].replace('.'+label_dict[k]['segment'],'')
                for k in label_dict:
                    if label_dict[k]['abbr'] is not None:
                        if 'gmwm2gether' in grouping:
                            if label_dict[k]['tissue'] is not None:
                                label_dict[k]['abbr'] = label_dict[k]['abbr'].replace('.'+label_dict[k]['tissue'],'')
                        if 'segmented' in grouping and 'lobe' not in grouping:
                            if label_dict[k]['segment'] is not None:
                                label_dict[k]['abbr'] = label_dict[k]['abbr'].replace('.'+label_dict[k]['segment'],'')
                        if 'hemisphere' in grouping:
                            if label_dict[k]['side'] is not None:
                                label_dict[k]['abbr'] = label_dict[k]['abbr'].replace('.'+label_dict[k]['side'],'')
            if df is not None:
                #if dataframe is passed, it will try to change the names of where the columns with Imperial are
                if isinstance(df,pd.DataFrame):
                    col = kwargs.get('col','index')
                    if col == 'index' or col == 'columns':
                        return df.rename({k:v['abbr'] for k,v in label_dict.items()},axis=col)
                    else:
                        df[col] = df[col].map({k:v['abbr'] for k,v in label_dict.items()})
                        return df
                elif isinstance(df,list):
                    return [label_dict[k]['abbr'] for k in df]
                else:
                    raise TypeError('df must be dataframe or list of strings')
            else:
                return label_dict

        def group_Imperial_volumes(df:pd.DataFrame,grouping:[list,tuple,str]=None,
                                   operation:str='sum',drop_duplicates:bool=True):
            """
            Group Imperial Volume by summing them.
            Use grouping scheme as defined in get_Imperial_legends
            
            NOTE: even if the df doesn't contain the necessary regions
            i.e., df was applied get_segments before grouped.
            it will still work.
            
            Parameters
            ----------
            df : pd.DataFrame
                The data Frame containing the Imperial volumes as columns.
            grouping : {'lobe','gmwm2gether','segmented','hemisphere'}, optional
                lobe- combine the regions by lobes (frontal, temporal, occipital, parietal)
                gmwm2gether- combine the same region- white and grey matter.
                segmented - combine the same region - anterior and posterior
                hemisphere - combine the same region - left and right. The default is None.
            operation: str. {'sum','mean'}
                how to group them. (mean can be used when generating positional info)
            drop_duplicates: bool
                Whether to drop the columns after grouping process.
                by default, the first column in the group list is retained as the total
                value of the rest of the values in the list and all values but the first one
                is removed.
                If False. All values in the grouped list will have the same values.
                This is needed when plotting the Brain Map. Otherwise only the first value will show.
            Returns
            -------
            new_df : pd.dataFrame
                grouped DataFrame.

            """
            if not isinstance(df,pd.DataFrame):
                raise TypeError('Needs pandas DataFrame')
            new_df = df.copy()
            grouped_volumes_dict = Volumes.Imperial.get_Imperial_legends(grouping)
            grouped_volumes_dict = {k:v['abbr'] for k,v in grouped_volumes_dict.items() if k in new_df.columns}#{'Imperial 1:'hipp.L'}
            unique_grouped_volumes_dict = defaultdict(list)
            for k,v in grouped_volumes_dict.items():
                unique_grouped_volumes_dict[v].append(k)
            for k,v in unique_grouped_volumes_dict.items():
                if k is None:
                    new_df = new_df.drop(columns = v)
                else:
                    if len(v) > 1:
                        if operation == 'sum':
                            new_df[v[0]] = df[v].sum(axis=1)
                        elif operation == 'mean':
                            new_df[v[0]] = df[v].mean(axis=1)
                        if drop_duplicates:
                            new_df = new_df.drop(columns=[i for i in df.columns if i in v[1:]])
                        else:
                            for col in v[1:]:
                                if col in df.columns:
                                    new_df[col] = new_df[v[0]]
            return new_df
            
                
                
    class AAL:    
        @staticmethod
        def group_AAL_volumes(df:pd.DataFrame,
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
        
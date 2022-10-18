from matplotlib.colors import ListedColormap
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
try:
    import data_exploration
except ModuleNotFoundError:
    from . import data_exploration
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm
from typing import List, Union, Optional
import nibabel as nib # used to do visualise brain maps
import copy
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib as mpl
from collections import defaultdict

# def nx_kamada_kawai_layout(test_graph):
#     '''
#     Input: requires the networkx graph object
#     '''
#     weights = nx.get_edge_attributes(test_graph, 'weight').values()
#     pos = nx.kamada_kawai_layout(test_graph)
#     node_hubs = [(node, degree) for node, degree in sorted(dict(test_graph.degree).items(
#     ), key=lambda item:item[1], reverse=True)][:5]  # sort dictionary by the values in the descending order
#     node_hubs_names = [node for node, degree in node_hubs]
#     labels = {}
#     for node in test_graph.nodes:
#         if node in node_hubs_names:
#             # set the node name as the key and the label as its value
#             labels[node] = node
#     # set the argument 'with labels' to False so you have unlabeled graph
#     nx.draw(test_graph, pos, width=list(weights), node_size=50,
#             node_color='lightgreen', with_labels=False)
#     # Now only add labels to the nodes you require
#     nx.draw_networkx_labels(test_graph, pos, labels,
#                             font_size=16, font_color='r')


# def save_in_npy(original_function, file_path):
#     def wrapper(*args, **kwargs):
#         result = original_function(*args, **kwargs)
#         return result
#     return wrapper


def plot_Linear_Reg(x: Union[np.ndarray, pd.DataFrame, pd.Series, str],
                    y: Union[np.ndarray, pd.DataFrame, pd.Series, str],
                    data: Optional[pd.DataFrame] = None,
                    hue: Optional[str] = None,
                    combined: Optional[bool] = False,
                    title: str = None,
                    xlabel: str = None,
                    ylabel: str = None,
                    ax = None,scaling = 'both', **figkwargs) -> None:
    """
    Fit linear regression, where y~x. calculates the pval and beta coefficient.
    
    You can use it to visualise across different populations and generate separate pval and beta coeficient for each population (hue) or for all of them combined
    Note: when you have two variables, the pearson's correlation coefficient is the same as the standardized beta coefs.
    Parameters
    ----------
    x : Union[np.ndarray, pd.DataFrame, pd.Series, str]
        value on x.
    y : Union[np.ndarray, pd.DataFrame, pd.Series, str]
        value on y.
    data : Optional[pd.DataFrame], optional
        if providing string x,y, then data will be the dataframe. The default is None.
    hue : Optional[str], optional
        separate data point by another value in the dataframe (e.g. cohort). It will calculate separate beta and p-value. The default is None.
    combined : Optional[bool], optional
        If use, calculate the total p-val and beta coefs. The default is False.
    title : str, optional
        Title of the graph. The default is None.
    xlabel : str, optional
         label on x axis. The default is None.
    ylabel : str, optional
        label on y axis. The default is None.
    axes : TYPE, optional
        if provided plt.subplots. The default is None.
    scaling : TYPE, optional
        whether to scale x and y. The default is 'both'.
    **figkwargs :
        linewdith: float
        markersize: float
        legend_loc {outside, inside}
        hide_CI=False
    Returns
    -------
    ax
        The ax plot.

    """
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    elif isinstance(x, str):
        x = data.loc[:, x].values
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values
    elif isinstance(y, str):
        y = data.loc[:, y].values
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if 'linewidth' not in figkwargs:
        figkwargs['linewidth'] = 1.5
    if 'markersize' not in figkwargs:
        figkwargs['markersize'] = 1.5
    if 'hide_CI' not in figkwargs:
        figkwargs['hide_CI'] = False
    
    def plotting(x, y, unique_label=None, combined=False, scaling=scaling):
        model, _ = data_exploration.MassUnivariate.mass_univariate(cont_independentVar_cols=x,
                                                                   dependentVar_cols=y,
                                                                   scaling=scaling)  # will perform standaridzation inside the function
        if scaling == 'both':
            x = StandardScaler().fit_transform(x)
            y = StandardScaler().fit_transform(y)
        elif scaling == 'x':
            x = StandardScaler().fit_transform(x)
        elif scaling == 'y':
            y = StandardScaler().fit_transform(y)
        y_pred = model.predict(sm.add_constant(x))
        predictions = model.get_prediction()
        df_predictions = predictions.summary_frame()
        sorted_x = np.argsort(x[:, 0])
        coefs = model.params.values[1]
        p_value = model.pvalues.values[1]
        
        if unique_label is None:
            beta_label = r'$\beta$=%0.03f, pval = %0.03f' % (coefs, p_value)
            
            if not combined:
                ax.plot(x[:, 0], y, '.', label='target',markersize=figkwargs['markersize'])
                ax.plot(x[sorted_x, 0], y_pred[sorted_x], '-', label=beta_label,linewidth=figkwargs['linewidth'])
            else:
                ax.plot(x[:, 0], y, 'o', label='total',alpha=.01,markersize=figkwargs['markersize'])
                handles, labels = ax.get_legend_handles_labels()
                ax.plot(x[sorted_x, 0], y_pred[sorted_x], '-',
                    label=beta_label, color=handles[len(handles)-1].get_color(),linewidth=figkwargs['linewidth'])
            if not figkwargs['hide_CI']:
                ax.fill_between(x[sorted_x, 0], df_predictions.loc[sorted_x, 'mean_ci_lower'], df_predictions.loc[sorted_x,
                                'mean_ci_upper'], linestyle='--', alpha=.1, color='crimson', label=unique_label)

        else:
            beta_label = r'$\beta$=%0.03f, pval = %0.03f' % (coefs, p_value)
            ax.plot(x[:, 0], y, '.', label=unique_label,markersize=figkwargs['markersize'])
            handles, labels = ax.get_legend_handles_labels()
            ax.plot(x[sorted_x, 0], y_pred[sorted_x], '-',
                    label=beta_label, color=handles[len(handles)-1].get_color(),linewidth=figkwargs['linewidth'])
            if not figkwargs['hide_CI']:
                ax.fill_between(x[sorted_x, 0], df_predictions.loc[sorted_x, 'mean_ci_lower'], df_predictions.loc[sorted_x,
                                'mean_ci_upper'], linestyle='--', alpha=.1, color='crimson')
    if ax is None:
        fig, ax = plt.subplots()
    if hue is None:
        plotting(x, y,scaling=scaling)
    else:
        data = data.reset_index(drop=True)
        unique_hues = data[hue].unique()
        for idx, unique_hue in enumerate(unique_hues):
            temp_data = data[data[hue] == unique_hue].index.to_list()
            plotting(x[temp_data], y[temp_data],
                     unique_label=unique_hue,scaling=scaling)
        if combined:
            plotting(x,y,combined=True,scaling=scaling)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if 'legend_loc' not in figkwargs:
        figkwargs['legend_loc'] = 'outside'
    if figkwargs['legend_loc'] == 'outside':
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc='lower left')
    ax.set_title(title)
    return ax

# def plot_correlation(x: Union[np.ndarray,pd.DataFrame,pd.Series,str],
#                      y: Union[np.ndarray,pd.DataFrame,pd.Series,str],
#                      data:pd.DataFrame=None,
#                      title:str=None, 
#                      xlabel:str=None, 
#                      ylabel:str=None,
#                      c:np.ndarray = None,
#                      cmap = 'jet',
#                      colorbar_label=None,
#                      scaling=None,
#                      ax=None):
#     if isinstance(x,(pd.Series,pd.DataFrame)):
#         x = x.values
#     elif isinstance(x,str):
#         if data is None:
#             raise ValueError('dataframe is missing')
#         x = data.loc[:,x].values
#     if isinstance(y,(pd.Series,pd.DataFrame)):
#         y = y.values
#     elif isinstance(x,str):
#         if data is None:
#             raise ValueError('dataframe is missing')
#         y = data.loc[:,y].values
#     if x.ndim == 1:
#         x = x.reshape(-1,1)
#     if y.ndim == 1:
#         y = y.reshape(-1,1)
#     if scaling == 'both':
#         x = StandardScaler().fit_transform(x)
#         y = StandardScaler().fit_transform(y)
#     elif scaling == 'x':
#         x = StandardScaler().fit_transform(x)
#     elif scaling == 'y':
#         y = StandardScaler().fit_transform(y)
        
#     lin_reg = LinearRegression()
#     lin_reg.fit(x,y)# best fit line
    
#     if c is not None:
#         plt.scatter(x,y,c=c,cmap=cmap)
#         plt.colorbar(label = colorbar_label)
#     else:
#         plt.scatter(x, y)
#     plt.plot(np.asarray(x), lin_reg.predict(
#         np.asarray(x).reshape(-1, 1)).reshape(-1),'-',color='orange')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
    # corr, p = pearsonr(x, y)
#     plt.figtext(0, 0, 'corr=%0.03f, pval=%0.03f' % (corr, p))

def draw_box_plots(df,
                   dependentVar=None,
                   independentVar=None,
                   threshold=None,
                   percentage=0.2,
                   ancestry_PCs=None,
                   ylabel=None,ax=None):
    """
    draw box plots by dividing the dataset into high and low risk
    """
    if not independentVar:
        measure = np.asarray(df[dependentVar])
    else:
        measure = data_exploration.MassUnivariate.adjust_covariates_with_lin_reg(np.asarray(
            df[dependentVar]), StandardScaler().fit_transform(df[independentVar]))

    high_risk, low_risk = data_exploration.divide_high_low_risk(data_exploration.MassUnivariate.adjust_covariates_with_lin_reg(
        np.asarray(df[threshold]), StandardScaler().fit_transform(df[ancestry_PCs])), low_perc=percentage, high_perc=percentage)

    measure_df = pd.DataFrame({'measure': measure.reshape(-1), 'risk': [
                              'high risk' if i in high_risk else 'low risk' if i in low_risk else np.nan for i in range(len(measure))]}).dropna()
    stats, pval = ttest_ind(measure_df[measure_df['risk'] == 'low risk']['measure'],
                            measure_df[measure_df['risk'] == 'high risk']['measure'], equal_var=True)
    low_risk_mean = measure_df[measure_df['risk']
                               == 'low risk']['measure'].mean(axis=0)
    high_risk_mean = measure_df[measure_df['risk']
                                == 'high risk']['measure'].mean(axis=0)

    sns.boxplot(x='risk', y='measure', data=measure_df,ax=ax)
    if not independentVar:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('corrected ' + ylabel)
    ax.text(0, 0, 't-test pval=%0.03f' % (pval))

class Brainmap:
    
    def __init__(self,atlas_file:Union[str,nib.nifti1.Nifti1Image],
                 cmap:str='Spectral',cmap_reversed:bool=False):
        if isinstance(atlas_file,str):
            self.atlas_file = nib.load(atlas_file)
        elif isinstance(atlas_file,nib.nifti1.Nifti1Image):
            self.atlas_file = atlas_file
        atlas_affine = self.atlas_file.affine
        if nib.aff2axcodes(atlas_affine) != ('R','A','S'): # if it is not RAS orientation, change to it.
            self.atlas_file = nib.as_closest_canonical(self.atlas_file)
        self.atlas = self.atlas_file.get_fdata()
        self.cmap = copy.copy(plt.cm.get_cmap(cmap))
        self.cmap.set_bad(alpha=0) # set transparency of np.nan numbers into 0
        if cmap_reversed:
            self.cmap = self.cmap.reversed()
    
    @staticmethod
    def get_edges(outline): # use in brain_segmentation
        """
        Use in the brain segmentation
        """
        #for each pixel colour the edges.
        edges = []
        rr,cc = np.nonzero(outline)
        switcheroo = lambda x,y: (y,x) # the imshow and the np.matrix have different coordinates
        def check_sides(r,c,outline):
            sides = {'top':False,'bottom':False,'left':False,'right':False}
            #the key is visualise that the 0,0 of the numpy matrix is upper left.
            if r == 0 or not outline[r-1,c]:
                # top edge
                sides['top'] = True
            if r == outline.shape[0] -1 or not outline[r+1,c]:
                # bottom edge
                sides['bottom'] = True
            if c == 0 or not outline[r,c-1]:
                # left edge
                sides['left'] = True
            if c == outline.shape[1]-1 or not outline[r,c+1]:
                # right edge
                sides['right'] = True
            return sides
        
        for r,c in zip(rr,cc):
            ul_coord = switcheroo(r,c) # upper left
            ll_coord = switcheroo(r+1,c) # lower left
            ur_coord = switcheroo(r,c+1) # upper right
            lr_coord = switcheroo(r+1,c+1)# lower right
            
            sides_dict = check_sides(r,c,outline)
            if sides_dict['top'] == True:
                edges.append([ul_coord,ur_coord])# top edge
            if sides_dict['bottom'] == True:
                edges.append([ll_coord,lr_coord])# bottom edge
            if sides_dict['left'] == True:
                edges.append([ul_coord,ll_coord])# left edge
            if sides_dict['right'] == True:
                edges.append([ur_coord,lr_coord])# right edge

        return np.array(edges)-0.5
    
    @classmethod
    def plot_segmentation(cls,
                         map_view:List=['all'],
                         atlas_slice:Union[int,list,dict]=None,
                         regions_to_hide:list=None,
                         plot_values:dict=None,
                         plot_values_threshold:float=None,
                         mask:dict=None,
                         fig:mpl.figure.Figure=None,
                         axes:Union[np.ndarray,List]=None,
                         colorbar:bool=False,
                         label_legend:dict=None,
                         legends:bool=True,
                         atlas_file:Union[str,nib.nifti1.Nifti1Image]=None,
                         cmap:str='Spectral',plot_orientation:bool=True,**figkwargs):
        """
        Plot the brain segmentation

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        map_view : List, optional
            {'all','sagittal','axial','coronal'}. The default is ['all'].
        atlas_slice: int,list, dict, optional
            dictionary keys: axial, coronal, sagittal, values = slice number
            if list: three numbers corresponds to axial, coronal, sagittal
            Choose which slice to show, else middle slice will be shown.
        regions_to_hide : list, optional
            Give the list of regions to hide (the regions must be list of intergers) The labels will be set to transparency 0. The default is None.
        plot_values : Union[dict], optional
            plot a color for each label denoting the strength of p-value or some other metrics.
            If provide a dictionary, the key will be the label, and the value will be the value. The default is None.
        plot_values_threshold:float, optional
            A number that can be applied to plot_values and set the ones that do not pass the threshold to np.nan
        mask : Union[dict], optional
            Can pass a mask, that after passing the plot_values threshold, instead of plotting the original plot_values, the mask will be plotted instead.
            E.g. I have p-values as plot_values, but I want to plot beta- coefficients that has p-value <0.05 instead. Here the dict of beta coef is the mask.
        fig : mpl.figure.Figure, optional
            Can specify the fig, if not fig will be created. The default is None.
        axes : Union[np.ndarray,List], optional
            Can provide the specific axes, list of axes. If not axes will be created. The default is None.
        colorbar : bool, optional
            If plot_values is specified, then can choose if to plot the colorbar. The default is False.
        label_legend : dict, optional
            If not using the plot_values, but you want to show the segmentation.
            Each color patch in the legend will correspond to the color in the plot
            Must provide a dictionary, where the key is interger denoting the label.
            And value is the string abbreviation. The default is None.
        atlas_file : Union[str,nib.nifti1.Nifti1Image], optional
            The path to the nifti file, or the nib.load(Nifti file). The default is None.
        cmap : str, optional
            The color scheme. The default is 'Spectral'.
        plot_orientation: bool, optional
            If you want to plot the axis in the RAS coordinate.
        **figkwargs : TYPE
            cmap_reversed = if you want to revese the cmap default False
            cb_orientation = {'horizontal','vertical'} if you want your change your colorbar orientation. Default horizontal next to the last axes.
            cb_title = str. name of the colorbar
            figsize = Default (20,10)
            outline_label_legends: bool. Default True. The outline is updated if the the same legend is found in two regions.
            outline_regions_to_hide: bool. Default True. The outline is not updated after using regions_to_hide. 
            label_legend_bbox_to_anchor:(-2.5, -1,0,0)
            label_legend_ncol : 6
            label_legend_loc: 'lower left'
            label_legend_fontsize: 'medium' or float
        Raises
        ------
        ValueError
            DESCRIPTION.
        AttributeError
            DESCRIPTION.

        Returns
        -------
        fig : TYPE
            The mpl.Figure either created or used.
        map_view_dict : TYPE
            a dictionary containing all atlas array, including 'im' plotted.
        """
        if 'cmap_reversed' not in figkwargs:
            figkwargs['cmap_reversed'] = False
        brain_map = cls(atlas_file,cmap,figkwargs['cmap_reversed'])
        brain_map.atlas[brain_map.atlas == 0] = np.nan # set the background to 0 transparency
        #the following original atlas are needed for the outline.
        atlas_slice_dict = {'axial':brain_map.atlas.shape[2]//2,
                         'coronal':brain_map.atlas.shape[2]//2,
                         'sagittal':brain_map.atlas.shape[2]//2}

        if isinstance(atlas_slice,int):
            atlas_slice_dict['axial'] = atlas_slice
            atlas_slice_dict['coronal'] = atlas_slice
            atlas_slice_dict['sagittal'] = atlas_slice
        elif isinstance(atlas_slice, list):
            if len(atlas_slice)!=3:
                raise ValueError('atlas slice must be 3, stands for axial, coronal, sagittal')
            atlas_slices = [slice_int if slice_int is not None else brain_map.atlas.shape[2]//2 for slice_int in atlas_slice]
            atlas_slice_dict = dict(zip(['axial','coronal','sagittal'],atlas_slices))
        elif isinstance(atlas_slice,dict):
            for key,value in atlas_slice.items():
                atlas_slice_dict[key] = value                
        
        #original atlases are used to delineate the outline
        original_axial_atlas = brain_map.atlas[:,:,atlas_slice_dict['axial']].copy()
        original_coronal_atlas = brain_map.atlas[:,atlas_slice_dict['coronal'],:].copy()
        original_sagittal_atlas = brain_map.atlas[atlas_slice_dict['sagittal'],:,:].copy()
                
        if label_legend is not None:
            unique_regions = np.unique(brain_map.atlas[~np.isnan(brain_map.atlas)]) #not labelling the np.nan values
            for region in unique_regions:
                if region not in label_legend:
                    brain_map.atlas[brain_map.atlas==region] = np.nan
            
            #check for unique legends- basically group the labels/ regions if the legend is the same. the original outline will still be shown
            unique_legends = set(label_legend.values()) 
            legend_label = {uniq_leg:[k for k in label_legend.keys() if label_legend[k] == uniq_leg] for uniq_leg in unique_legends}
            for legend,label in legend_label.items():
                if len(label)>1:
                    for region in label:
                        brain_map.atlas[brain_map.atlas==region] = label[0]
            
            if 'outline_label_legends' not in figkwargs:
                figkwargs['outline_label_legends'] = True
            if figkwargs['outline_label_legends']:
                original_axial_atlas = brain_map.atlas[:,:,atlas_slice_dict['axial']].copy()
                original_coronal_atlas = brain_map.atlas[:,atlas_slice_dict['coronal'],:].copy()
                original_sagittal_atlas = brain_map.atlas[atlas_slice_dict['sagittal'],:,:].copy()
        
        # the following regions need to be hide (but outline will still be shown)
        if regions_to_hide is not None:
            for region in regions_to_hide:
                brain_map.atlas[brain_map.atlas == region] = np.nan
                if 'outline_regions_to_hide' not in figkwargs:
                    figkwargs['outline_regions_to_hide'] = True
                if not figkwargs['outline_regions_to_hide']:
                    original_axial_atlas = brain_map.atlas[:,:,atlas_slice_dict['axial']].copy()
                    original_coronal_atlas = brain_map.atlas[:,atlas_slice_dict['coronal'],:].copy()
                    original_sagittal_atlas = brain_map.atlas[atlas_slice_dict['sagittal'],:,:].copy()
            
                
                
        if plot_values is not None:
            if not isinstance(plot_values,dict):
                raise TypeError('plot_values needs to be a dictionary, where keys are the label, and values are the plot values')
            else:
                unique_regions = np.unique(brain_map.atlas[~np.isnan(brain_map.atlas)])
                if plot_values_threshold is not None:
                    if mask is not None:
                        if not isinstance(mask,dict):
                            raise TypeError('mask needs to be a dictionary, where keys are the label, and values are the plot values')
                        plot_values = {indx:mask[indx] if (indx in plot_values.keys()) and (plot_values[indx]>plot_values_threshold) and (indx in mask.keys()) else np.nan for indx in unique_regions}
                    else:
                        plot_values = {indx:plot_values[indx] if (indx in plot_values.keys()) and (plot_values[indx]>plot_values_threshold) else np.nan for indx in unique_regions}                  
                else:
                    plot_values = {indx:plot_values[indx] if indx in plot_values.keys() else np.nan for indx in unique_regions}
                for region,value in plot_values.items():
                    brain_map.atlas[brain_map.atlas == region] = value
        
        if 'figsize' not in figkwargs:
            figkwargs['figsize'] = (22,10)
        
        if axes is None:
            if 'all' in map_view:
                axes = 3
            else:
                axes = len(map_view)
            fig, axes  = plt.subplots(1,axes,figsize=figkwargs['figsize'])
            try:
                len(axes)
            except TypeError:
                axes = np.asarray([axes]) # because when I zip in zip(map_view,axes) I can't zip len of 1? and len(axessubplot) doesnt return anything :)))
        elif isinstance(axes,(list,np.ndarray)):
            if 'all' in map_view:
                if len(axes) != 3:
                    raise ValueError('need 3 axes')
            else:
                if len(map_view) != len(axes):
                    raise ValueError('number of map_view does not match number of axes provided')
            if (fig is None) and (plot_values is not None) and (colorbar is not None):
                raise AttributeError('Need fig element to plot the colorbar')
        
        #
        axial_atlas = brain_map.atlas[:,:,atlas_slice_dict['axial']].copy()
        coronal_atlas = brain_map.atlas[:,atlas_slice_dict['coronal'],:].copy()
        sagittal_atlas = brain_map.atlas[atlas_slice_dict['sagittal'],:,:].copy()

        map_view_dict = defaultdict(dict)
        if 'all' in map_view:
            map_view = ['axial','coronal','sagittal']
        for view,ax in zip(map_view,axes):
            map_view_dict[view]['ax'] = ax
            if view == 'axial':
                map_view_dict[view]['atlas'] = axial_atlas
                map_view_dict[view]['original_atlas'] = original_axial_atlas
            elif view == 'coronal':
                map_view_dict[view]['atlas'] = coronal_atlas
                map_view_dict[view]['original_atlas'] = original_coronal_atlas
            elif view == 'sagittal':
                map_view_dict[view]['atlas'] = sagittal_atlas
                map_view_dict[view]['original_atlas'] = original_sagittal_atlas
        
        #plot the images
        vmin = np.min([map_view_dict[view]['atlas'][~np.isnan(map_view_dict[view]['atlas'])].min() for view in map_view_dict.keys()])
        vmax = np.max([map_view_dict[view]['atlas'][~np.isnan(map_view_dict[view]['atlas'])].max() for view in map_view_dict.keys()])

        for view,ax in map_view_dict.items():
            map_view_dict[view]['im'] = map_view_dict[view]['ax'].imshow(np.rot90(map_view_dict[view]['atlas']),vmin=vmin,vmax=vmax,cmap=brain_map.cmap)
    
            #plot the outlines
            temp_original_atlas = map_view_dict[view]['original_atlas']
            for unique_label in np.unique(temp_original_atlas[~np.isnan(temp_original_atlas)]):
                #if it is that label, draw the outline
                temp_outline_atlas = temp_original_atlas.copy()
                temp_outline_atlas[temp_outline_atlas != unique_label] = 0 # if it is not that label, set the pixel to 0
                temp_outline_atlas[temp_outline_atlas == unique_label] = 1 # basically draw the border where there is pixel value 1
                temp_line = LineCollection(cls.get_edges(np.rot90(temp_outline_atlas)), lw=1, color='k')
                map_view_dict[view]['ax'].add_collection(temp_line)
        
        if 'cb_orientation' not in figkwargs:
            figkwargs['cb_orientation'] = 'vertical'
        #add the colorbar?
        if colorbar: # the colorbar is added to the last im
            if plot_values is None:
                raise ValueError('need plot value to have colorbar')
            cb = fig.colorbar(map_view_dict[list(map_view_dict)[-1]]['im'],ax = map_view_dict[list(map_view_dict)[-1]]['ax'], orientation = figkwargs['cb_orientation'])
            if 'cb_title' not in figkwargs:
                figkwargs['cb_title'] = None
            if figkwargs['cb_orientation'] == 'vertical':
                cb.ax.set_ylabel(figkwargs['cb_title'],rotation=90,fontsize=12,fontweight='bold')
            elif figkwargs['cb_orientation'] == 'horizontal':
                cb.ax.set_xlabel(figkwargs['cb_title'],rotation=0,fontsize=12,fontweight='bold')
        
        if 'label_legend_bbox_to_anchor' not in figkwargs:
            figkwargs['label_legend_bbox_to_anchor'] = (-2.5, -1,0,0)
        if 'label_legend_ncol' not in figkwargs:
            figkwargs['label_legend_ncol'] = 6
        if 'label_legend_loc' not in figkwargs:
            figkwargs['label_legend_loc'] = 'lower left'
        if 'label_legend_fontsize' not in figkwargs:
            figkwargs['label_legend_fontsize'] = 'medium'
        if label_legend is not None:
            #to plot legends?
            if legends:
                #get the unique labels
                values = np.unique(np.concatenate([map_view_dict[view]['atlas'].ravel() for view in map_view]))
                values = values[~np.isnan(values)] #not labelling the np.nan values
                # get the colors of the values, according to the 
                # colormap used by imshow
                temp_im = map_view_dict[list(map_view_dict)[0]]['im']
                colors = [temp_im.cmap(temp_im.norm(value)) for value in values]
                patches = [mpatches.Patch(color=colors[idx], label=label_legend[int(i)]) for idx, i in enumerate(values) if i in label_legend]
                plt.legend(handles=patches, 
                           bbox_to_anchor=figkwargs['label_legend_bbox_to_anchor'], 
                           loc=figkwargs['label_legend_loc'],
                           ncol=figkwargs['label_legend_ncol'],
                           fontsize = figkwargs['label_legend_fontsize'],
                           frameon=False)

        for view in map_view:
            sns.despine(bottom=True,left=True,right=True)
            map_view_dict[view]['ax'].set_xticks([])
            map_view_dict[view]['ax'].set_yticks([])
            
        if plot_orientation:
            for view,ax in map_view_dict.items():
                # map_view_dict[view]['ax'].axhline(map_view_dict[view]['atlas'].shape[1]//2,alpha=0.1)
                # map_view_dict[view]['ax'].axvline(map_view_dict[view]['atlas'].shape[0]//2,alpha=0.1)
                if view == 'axial':
                    map_view_dict[view]['ax'].text(0,map_view_dict[view]['atlas'].shape[1]//2 + 2, 'L',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]//2 + 2, 0, 'A',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]-10,map_view_dict[view]['atlas'].shape[1]//2+2,'R',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]//2,map_view_dict[view]['atlas'].shape[1]-10,'P',fontsize=15,fontweight='bold')
                elif view == 'coronal':
                    map_view_dict[view]['ax'].text(0,map_view_dict[view]['atlas'].shape[1]//2 + 2, 'L',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]//2 + 2, 0, 'S',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]-10,map_view_dict[view]['atlas'].shape[1]//2+2,'R',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]//2,map_view_dict[view]['atlas'].shape[1],'I',fontsize=15,fontweight='bold')
                elif view == 'sagittal':
                    map_view_dict[view]['ax'].text(0,map_view_dict[view]['atlas'].shape[1]//2 + 2, 'P',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]//2 + 2, 0, 'S',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]-10,map_view_dict[view]['atlas'].shape[1]//2+2,'A',fontsize=15,fontweight='bold')
                    map_view_dict[view]['ax'].text(map_view_dict[view]['atlas'].shape[0]//2,map_view_dict[view]['atlas'].shape[1]-10,'I',fontsize=15,fontweight='bold')
        
        return fig, map_view_dict
        
    
    @classmethod
    def get_ROIs_coordinates(cls,
                             atlas_file:Union[str,nib.nifti1.Nifti1Image]):
        """
        Get ROIs coordinate where the x,y,z are the voxel indices.
        Parameters
        ----------
        atlas_file : Union[str,nib.nifti1.Nifti1Image]
            pathway to nifti file or the nib.load nifti file.

        Returns
        -------
        ROIs_coord : dict
            dictionary where for the key is label, and value is the x,y,z coordinates.
        """
        brain_map = Brainmap(atlas_file)
        brain_map.atlas[np.isnan(brain_map.atlas)] = 0 # if nan exist set to 0
        ROIs_coord = defaultdict(list)
        for label in np.unique(brain_map.atlas):
            if label != 0:
                ROIs_coord[label] = [coord.mean() for coord in np.where(brain_map.atlas == label)]
        ROIs_coord = pd.DataFrame(ROIs_coord).T
        ROIs_coord.reset_index(inplace=True)
        ROIs_coord.columns = ['Label','X','Y','Z']
        return ROIs_coord

        
class Geneset:

    @staticmethod
    def create_heatmap(gene_set_table:pd.DataFrame,
                       genes_set_column:str = 'GeneSet',
                       genes_list_column:str = 'genes',
                       gene_table:pd.DataFrame = None,
                       top:int = 20,
                       ordered_by:str='P_Frontal_lobe_WM',
                       gene_table_gene_name:str='Genes_Name') -> List[np.array]:
        """
        Create a heatmap of gene set analysis, where the x-axis is the list of genes, and y-axis is the gene set list.
        Args:
            gene_set_table (pd.DataFrame): enrichment result in table format
            |Geneset|P-value for phenotype 1| P-value for phenotype 2|List of genes| etc.
            genes_set_column (str, optional): _description_. Defaults to 'GeneSet'.
            genes_list_column (str, optional): _description_. Defaults to 'genes'.
            gene_table (pd.DataFrame, optional): table containing rows of genes (or SNPs) and columns of P-value associated with 1 or 2 phenotypes (usually 1 for variable of interest and 1 taken directly from GWAS schizophrenia). Defaults to None.
        Note:
        The following genes name in the FUMA Gene-sets do not correspond with genes name in gene-table
        ['ADGRV1' if gene == 'GPR98' else 'ADGRB3' if gene == 'BAI3' else gene for gene in FUMA_gene_sets]
        
        Returns:
            heatmap: binary np.array, the x axis = list of genes, y axis = gene sets, 1 where there is a presence of the gene in gene set.
            all_genes = genes names on the x-axis
            gene_sets = genes set names on the y-axis
        """
        def heatmap_array(genesets:List[str],
                          genelist:List[str],
                          geneset_dict:dict) -> np.array:
            """
            Attributes:
                genesets = list of gene sets
                genelist = list of genes
                geneset_dict = dictionary where keys are genesets, and values are gene lists
            Return
                heatmap: binary np.array, the x axis = list of genes, y axis = gene sets, 1 where there is a presence of the gene in gene set.
                all_genes = genes names on the x-axis
                gene_sets = genes set names on the y-axis
            """
            heatmap = np.zeros((len(genesets),len(genelist)))
            for row,gene_set in enumerate(genesets):
                for column, gene in enumerate(genelist):
                    if gene in geneset_dict[gene_set]:
                        heatmap[row,column] = 1
            # new_idx = [idx for idx,gene in enumerate(genelist) if gene in list(geneset_dict.values())[0]]
            # old_idx = [idx for idx in range(len(genelist)) if idx not in new_idx]
            # new_idx = new_idx + old_idx
            # heatmap = heatmap[:,new_idx]
            # genelist = [genelist[idx] for idx in new_idx]
            return heatmap, genelist, list(geneset_dict.keys())
        
        gene_set_dict = {k:v.split(':') for k,v in zip(gene_set_table[genes_set_column].tolist(),gene_set_table[genes_list_column].tolist())}
        all_genes = [gene for gene_set in gene_set_dict.values() for gene in gene_set] #all of the genes found in the enrichment result table
        all_genes = list(set(all_genes)) # get the unique values
        all_genes_sets = gene_set_dict.keys()
        if top is not None:
            if ordered_by is not None:
                if gene_table is None:
                    raise ValueError('Must provide gene table')
                gene_table.replace('ADGRV1','GPR98',inplace=True)
                gene_table.replace('ADGRB3','BAI3',inplace = True)
        
                gene_table_considered = gene_table[gene_table[gene_table_gene_name].isin(all_genes)].sort_values(by=ordered_by)
                
                gene_table_considered = gene_table_considered.drop_duplicates(gene_table_gene_name,keep='first') # if there is a duplicate in the gene name keep the first (lowest p-value).
            
                gene_list_considered = gene_table_considered[gene_table_gene_name].tolist()[:top] # the gene_list_to_considered
                
                return heatmap_array(all_genes_sets,
                                     gene_list_considered,
                                     gene_set_dict)
            else:
                heatmap,gene_list_considered, gene_sets = heatmap_array(all_genes_sets,
                                        all_genes,
                                        gene_set_dict)
                heatmap = heatmap[:,:top]
                gene_list_considered = gene_list_considered[:top]
                return heatmap, gene_list_considered, gene_sets
        else:
            return heatmap_array(all_genes_sets,
                                 all_genes,
                                 gene_set_dict)
    
    @staticmethod
    def visualise_enrichment_p_value(gene_set_table:pd.DataFrame,
                                     x:str='adjP',
                                     y:str = 'GeneSet',
                                     colour_by:str = 'Proportion',
                                     xlabel:str = None,
                                     ylabel:str = None,
                                     ax:np.array=None,
                                     cbar_ax:np.array=None):
        norm = plt.Normalize(gene_set_table[colour_by].min(),
                             gene_set_table[colour_by].max())
        sm = plt.cm.ScalarMappable(cmap='Reds',norm = norm)
        sm.set_array([])
        
        g = sns.barplot(x=-np.log10(gene_set_table[x]), y=y, hue=colour_by, data=gene_set_table,palette='Reds', 
                        dodge=False,ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend_.remove()
        cbar = ax.figure.colorbar(sm,ax = ax)
        cbar.set_label(colour_by,fontsize=15)
        ax.set_yticks([])        
        return ax
    @staticmethod    
    def visualise_heatmap(heatmap:np.array,
                          all_genes:List[str],
                          gene_sets:List[str],
                          ax:np.array,**figkwargs)->None:
        """Visualise the gene-set heatmap

        Args:
            heatmap (np.array): the heatmap generated by GeneSet.create_heatmap
            all_genes (List[str]): the list of genes generated by GeneSet.create_heatmap
            ax (np.array): the plt.Ax to plot on.
        """
        ax.imshow(heatmap,vmin=0, vmax=1,aspect='equal',cmap='Blues')
        ax.set_xticks(np.arange(0,len(all_genes),1))
        ax.set_yticks(np.arange(0,len(gene_sets),1))
        
        ax.set_xticks(np.arange(-.5, len(all_genes), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(gene_sets), 1), minor=True)
        
        ax.set_xticklabels(all_genes,rotation=45,fontsize=15)
        ax.set_yticklabels(gene_sets,rotation=45,fontsize=15)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        return ax
    @staticmethod
    def visualise_gene_p_value(gene_table:pd.DataFrame,
                               all_genes:List[str],
                               gene_table_gene_name:str='Genes_Name',
                               ordered_by:str='P_Frontal_lobe_WM',
                               coloured_by:str='P_schizo',
                               bar_number:str='N_SNP',
                               p_threshold:float=5e-08,
                               ax:np.array=None,
                               cbar_ax=None)->None:
        """Visualise the p-value of the genes in genes list.

        Args:
            gene_table (pd.DataFrame): _description_
            all_genes (List[str]): _description_
            ordered_by (str,optional): the p-value column name to order by
            coloured_by (str, optional): the p-value column name to colour the bar. Defaults to P_schizo.
            bar_number (str, optional): the column name of number of SNPs in the gene. Defaults to N_SNPs.
            p_threshold (float, optional): _description_. Defaults to None.
            ax: the plt.Axes
        """
        gene_table = gene_table[gene_table[gene_table_gene_name].isin(all_genes)]
        gene_table = gene_table.sort_values(by=ordered_by).drop_duplicates(gene_table_gene_name,keep='first')# order and drop the duplicates
        gene_table[gene_table_gene_name] = pd.Categorical(gene_table[gene_table_gene_name],
                                                          categories=all_genes,
                                                          ordered=True)
        gene_table = gene_table.sort_values(by=gene_table_gene_name).reset_index(drop=True)
        bar_plot_y_axis = -np.log10(gene_table[ordered_by].tolist())
        
        g = sns.barplot(x=gene_table_gene_name,
                        y=bar_plot_y_axis,
                        data=gene_table,
                        ax=ax)
        
        if coloured_by is not None:
            bar_plot_colour_by = -np.log10(gene_table[coloured_by].tolist())
            if p_threshold is not None:
                bar_clrs = ['grey' if x < -np.log10(p_threshold) else 'red' for x in bar_plot_colour_by]
                for patch,bar_clr in zip(g.patches,bar_clrs):
                    patch.set_color(bar_clr)
                    
            else:
                pal = sns.color_palette('coolwarm',len(bar_plot_colour_by))
                rank = bar_plot_colour_by.argsort().argsort()
                index = gene_table.index.values
                my_cmap = ListedColormap(pal)
                norm = plt.Normalize(0, bar_plot_colour_by.max())
                sm = plt.cm.ScalarMappable(cmap=my_cmap,norm=norm)
                sm.set_array([])
                palette = np.array(pal)
                for patch,bar_clr in zip(g.patches,palette[rank[index]]):
                    patch.set_color(bar_clr)
                if cbar_ax is not None:
                    cbar = g.figure.colorbar(sm,cax = cbar_ax)
                    cbar.set_label('$-log_{10}(SNP p-value) SCZ $',fontsize=15)

        if bar_number is not None:                   
            for patch,label in zip(g.patches,gene_table.loc[:,bar_number].tolist()):
                if label > 1:
                    ax.annotate(label,
                                (patch.get_x()*1.005,
                                patch.get_height()*1.005),fontsize = 15)
        
        ax.set_xticks([])
        ax.set_xlabel('') 
        
        ax.set_ylabel('$-log_{10}(SNP p-value) FL_{WM}.R$',fontsize=15)
        return ax
        
        
        
        
    
def visualise_heatmap(df,
                      ax,
                      xlabel:str=None,
                      ylabel:str=None,
                      rotation=0):
    
    heatmap = df.copy()
    g = ax.imshow(heatmap,cmap='jet')
    ax.set_xticks(np.arange(0,heatmap.shape[1],1))
    ax.set_yticks(np.arange(0,heatmap.shape[0],1))
    
    ax.set_xticks(np.arange(-.5, heatmap.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, heatmap.shape[0], 1), minor=True)
    
    ax.set_xticklabels(xlabel,rotation=rotation,fontsize=12)
    ax.set_yticklabels(ylabel,rotation=rotation,fontsize=12)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    return g
    
    
    
        
    
    

    








        
        
from matplotlib.colors import ListedColormap
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from . import data_exploration
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm
from typing import List, Union, Optional


def nx_kamada_kawai_layout(test_graph):
    '''
    Input: requires the networkx graph object
    '''
    weights = nx.get_edge_attributes(test_graph, 'weight').values()
    pos = nx.kamada_kawai_layout(test_graph)
    node_hubs = [(node, degree) for node, degree in sorted(dict(test_graph.degree).items(
    ), key=lambda item:item[1], reverse=True)][:5]  # sort dictionary by the values in the descending order
    node_hubs_names = [node for node, degree in node_hubs]
    labels = {}
    for node in test_graph.nodes:
        if node in node_hubs_names:
            # set the node name as the key and the label as its value
            labels[node] = node
    # set the argument 'with labels' to False so you have unlabeled graph
    nx.draw(test_graph, pos, width=list(weights), node_size=50,
            node_color='lightgreen', with_labels=False)
    # Now only add labels to the nodes you require
    nx.draw_networkx_labels(test_graph, pos, labels,
                            font_size=16, font_color='r')


def save_in_npy(original_function, file_path):
    def wrapper(*args, **kwargs):
        result = original_function(*args, **kwargs)
        return result
    return wrapper


def plot_Linear_Reg(x: Union[np.ndarray, pd.DataFrame, pd.Series, str],
                    y: Union[np.ndarray, pd.DataFrame, pd.Series, str],
                    data: Optional[pd.DataFrame] = None,
                    hue: Optional[str] = None,
                    combined: Optional[bool] = False,
                    title: str = None,
                    xlabel: str = None,
                    ylabel: str = None,
                    axes = None,scaling = 'both', **figkwargs) -> None:
    """[Plot Linear Regression]

    Args:
        x (np.ndarray): [array -like independent var]
        y (np.ndarray): [array -like dependent var]
        title (str, optional): [description]. Defaults to None.
        xlabel (str, optional): [description]. Defaults to None.
        ylabel (str, optional): [description]. Defaults to None.
    Return:
        Plot
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
    def plotting(x, y, unique_label=None, combined=False, scaling=scaling):
        model, _ = data_exploration.MassUnivariate.mass_univariate(cont_independentVar_cols=x,
                                                    dependentVar_cols=y,scaling=scaling)  # will perform standaridzation inside the function
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
        
        if not unique_label:
            beta_label = r'$\beta$=%0.03f, pval = %0.03f' % (coefs, p_value)
            if not combined:
                ax.plot(x[:, 0], y, '.', label='target',markersize=figkwargs['markersize'])
                ax.plot(x[sorted_x, 0], y_pred[sorted_x], '-', label=beta_label,linewidth=figkwargs['linewidth'])
            else:
                ax.plot(x[:, 0], y, 'o', label='total',alpha=.01,markersize=figkwargs['markersize'])
                handles, labels = ax.get_legend_handles_labels()
                ax.plot(x[sorted_x, 0], y_pred[sorted_x], '-',
                    label=beta_label, color=handles[len(handles)-1].get_color(),linedwith=figkwargs['linedwidth'])
            ax.fill_between(x[sorted_x, 0], df_predictions.loc[sorted_x, 'mean_ci_lower'], df_predictions.loc[sorted_x,
                            'mean_ci_upper'], linestyle='--', alpha=.1, color='crimson', label=unique_label)
            # plt.figtext(0, 0, r'$\beta$=%0.03f, pval = %0.03f' %
            #             (coefs, p_value))

        else:
            beta_label = r'$\beta$=%0.03f, pval = %0.03f' % (coefs, p_value)
            ax.plot(x[:, 0], y, '.', label=unique_label,markersize=figkwargs['markersize'])
            handles, labels = ax.get_legend_handles_labels()
            ax.plot(x[sorted_x, 0], y_pred[sorted_x], '-',
                    label=beta_label, color=handles[len(handles)-1].get_color(),linewidth=figkwargs['linewidth'])
            ax.fill_between(x[sorted_x, 0], df_predictions.loc[sorted_x, 'mean_ci_lower'], df_predictions.loc[sorted_x,
                            'mean_ci_upper'], linestyle='--', alpha=.1, color='crimson')
    if not axes:
        fig, ax = plt.subplots()
    else:
        ax = axes
    if not hue:
        plotting(x, y)
    else:
        data = data.reset_index(drop=True)
        unique_hues = data[hue].unique()
        scalerx = StandardScaler().fit(x)
        x = scalerx.transform(x)
        scalery = StandardScaler().fit(y)
        y = scalery.transform(y)
        for idx, unique_hue in enumerate(unique_hues):
            temp_data = data[data[hue] == unique_hue].index.to_list()
            plotting(x[temp_data], y[temp_data],
                     unique_label=unique_hue,scaling=False)
        if combined:
            plotting(x,y,combined=True)
            

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if 'legend_loc' not in figkwargs:
        figkwargs['legend_loc'] = 'outside'
    if figkwargs['legend_loc'] == 'outside':
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc='lower left')
    ax.set_title(title)


def plot_correlation(x: List[np.ndarray],
                     y: np.ndarray,
                     title=None, xlabel=None, ylabel=None):
    lin_reg = LinearRegression()
    lin_reg.fit(np.asarray(x).reshape(-1, 1), np.asarray(y))
    plt.plot(x, y, '.')
    plt.plot(np.asarray(x), lin_reg.predict(
        np.asarray(x).reshape(-1, 1)).reshape(-1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    corr, p = pearsonr(x, y)
    plt.figtext(0, 0, 'corr=%0.03f, pval=%0.03f' % (corr, p))


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
    
    
    
        
    
    
    
    








        
        
"""graph_analysis.py
This contains the function to perform network statistics
"""
from random import choice,choices
from functools import partial
import numpy as np
import bct
import pandas as pd
import networkx as nx
from networkx import from_numpy_array
from networkx.algorithms.efficiency_measures import global_efficiency, local_efficiency
from networkx.algorithms.cluster import clustering
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length

from . import data_preprocessing

# def check_the_network_measurements(original_func,file_path):
#     """
#     Perform the checking of network measurements.
#     """
#     def wrapper(*args,**kwargs):
#         if inspect.isgeneratorfunction(original_func):        
#             result=next(original_func(*args,**kwargs))
#         else:
#             result=original_func(*args,**kwargs)
#         return result
#     return wrapper

def create_a_graph(Graph,edge_list,weight_list=None):
    """
    Create a networkx Graph with the specified nodes and weight list
    Arg:
        Graph (networkx Graph): an empty networkx Graph
        edge_list=list of tuples, where the first element is the source, and 
            the second is the target node.
        weight_list=list of weights for each of the corresponding edge.
    Output:
        Graph (networkx Graph)
    """
    source_list=[]
    target_list=[]
    for edge in edge_list:
        source_list.append(edge[0])
        target_list.append(edge[1])
    edges_df=pd.DataFrame(
        {
            "source":source_list,
            "target":target_list,
            "weight":weight_list
        }
    )
    Graph=nx.from_pandas_edgelist(edges_df,edge_attr=True)
    return Graph


def binarize_matrix_based_on_sparsity_threshold(correlation_matrix,percentage):
    """
    The connections are first ranked based on their weights (either diffusion 
        value or corrleation coefficients) and then we binarize the matrix so 
        that x% of the highest connections= 1 and the rest 0.
    Arg:
        corrmatrix(np.array): square matrix
        percentage (float): 0 to 1. percentage of x
        bins= bins in the histogram, the more bins, the more accurate will be 
            the binarization.
    Output:
        binarized_corrmatrix: the binarized matrix. square matrix
    
    """
    corrmatrix=correlation_matrix.copy()
    connections=data_preprocessing.lower_triangle(corrmatrix)
    connections_sorted=np.sort(abs(connections))[::-1]#high_to_low
    threshold=connections_sorted[int(len(connections)*percentage)-1]
    connections=np.where(abs(connections)>=threshold,float(1),float(0))
    binarized_corrmatrix=data_preprocessing.reverse_lower_triangle(connections,corrmatrix.shape[0])
    return binarized_corrmatrix

def yield_perm_matrices_volumetric_data(X,perm_run,network_sparisity,diffusion_data=False):
    """
    Perform non-parametric permutation testing (Shi et al., 2012)
        First, a network property measure (e.g., efficiency, clustering,
        small-worldness) is calculated separately for high-risk and low-risk
        neonates. Then randomly assign the regional cortical thickness measures
        of each subject to either groups. Correlation matrix was recomputed for
        each of the 2 randomized groups and a new value was obtained. This was
        repeated 1000 times and significance is reached if less than 5
        percentile of between group difference in the permutation distribution
        was greater than the observed group difference.
    Args:
        X (np.array): the group volumetric or diffusion data
        network_sparsity (list): list of network sparsity thresholds
        diffusion_data (Boolean): if X is diffusion data.
            If True (default False), instead of corr matrix, the mean connectivity value for 
            set of random low risk matrices, and high risk matrices in a
            shape of n x m where n is network sparsity thresholds and m is
            regions.

    """
    for _ in range(perm_run):
        new_y=np.asarray(choices([0,1],k=X.shape[0]))
        
        if diffusion_data:
            X_mean_connectivity_low_risk=np.mean(X[new_y==0],axis=0)# here the X_mean has a data shape of nx p, where p is number of unique features.
            X_mean_connectivity_high_risk=np.mean(X[new_y==1],axis=0)
            matrix_perm_low_risk=data_preprocessing.reverse_lower_triangle(X_mean_connectivity_low_risk)
            matrix_perm_high_risk=data_preprocessing.reverse_lower_triangle(X_mean_connectivity_high_risk)
        
        else:
            matrix_perm_low_risk=abs(np.corrcoef(X[new_y==0,:],rowvar=False))
            np.fill_diagonal(matrix_perm_low_risk,0)
            
            matrix_perm_high_risk=abs(np.corrcoef(X[new_y==1,:],rowvar=False))
            np.fill_diagonal(matrix_perm_high_risk,0)
        
        binarized_perm_low_risk_matrices=np.asarray([
            data_preprocessing.lower_triangle(
            binarize_matrix_based_on_sparsity_threshold(matrix_perm_low_risk,threshold/100,bins=100)) for threshold in network_sparisity])
        
        binarized_perm_high_risk_matrices=np.asarray([
            data_preprocessing.lower_triangle(
            binarize_matrix_based_on_sparsity_threshold(matrix_perm_high_risk,threshold/100,bins=100)) for threshold in network_sparisity])
        
        yield binarized_perm_low_risk_matrices,binarized_perm_high_risk_matrices

def generate_random_matrix(matrix):
    """
    Generate randomized graph similar to Maslov and Sneppen (2002)
        Randomized version of the graph is constructed by randomly reshuffling
        links, while keeping the in- and out- degree of each node constant.
        
        First, randomly select a pair of edges, (i1,j1) and (i2,j2). The two
        edges are rewired in such a way that i1 is connected to j2, while j1
        is connected to i2. However, in case one or both of these new links
        already exist in the network, this step is aborted and a new pair of
        edges is selected. This is to prevent the appearance of multiple edges
        connecting the same pair of nodes.
        
    Arguments:
        matrix: square undirected matrix (binarized)
    Returns:
        a new randomized matrix as explained in notes.
    Note:
        1. mask the upper triangle of the matrix to -1 and diagonal so that
            edges will be from the lower triangle only (so we do not pick the same edge twice. e.g., ab - ba ->)
        2. pick a random edge from the list. And for each node lists the open
            ends (e.g. for edge ab, both a and b is not connected to c and d.)
        3. Then check which of the open ended nodes form an edge and randomly
            choose one of them as the second edge.
        4. rewire the edges
        5. update the original matrix and repeat.
    """
    original_matrix=matrix.copy()
    np.fill_diagonal(original_matrix,9999)
    #make a list of all edges, choose the pos where the lower triangle, below the diagonal equals 1.
    edges_list=list(zip(*np.where(np.tril(original_matrix,k=-1)==1)))
    #randomly pick 1 of them
    for _ in range(len(edges_list)):
        try:
            (i1,j1)=choice(edges_list)
            #for selected edge, see where each end can connect to
            alli1holes=np.where(original_matrix[i1,:]==0)[0]
            allj1holes=np.where(original_matrix[j1,:]==0)[0]

            #we can only use edges with connection to neither node
            iintersect=np.intersect1d(alli1holes,allj1holes)
            #find which of these nodes are connected (the lower triangle)
            possible_second_edge=list(zip(*np.where(np.tril(original_matrix[iintersect[:,None],iintersect],k=-1)==1)))
            #if there is an edge to switch
            if len(possible_second_edge)!=0:
                edge2=choice(possible_second_edge)
                
                i2=iintersect[edge2[0]]
                j2=iintersect[edge2[1]]
                #update the original edges
                original_matrix[i1,j1]=original_matrix[j1,i1]=0
                original_matrix[i2,j2]=original_matrix[j2,i2]=0
                #update the rewired edges
                original_matrix[i1,j2]=original_matrix[j2,i1]=1
                original_matrix[i2,j1]=original_matrix[j1,i2]=1
                
                #update the edge_list
                edges_list.remove((i1,j1))
                edges_list.remove((i2,j2))

                if j2>i1:#important because we are using the lower triangle, where the index is (higher number, lower number)
                    edges_list.append((j2,i1))
                else:
                    edges_list.append((i1,j2))
                if j1>i2:
                    edges_list.append((j1,i2))
                else:
                    edges_list.append((i2,j1))
            
        except IndexError: # Cannot choose from an empty sequence
            break
    np.fill_diagonal(original_matrix,0)
    return original_matrix

def calculate_network_metrics_random_volumetric_data(matrix,n_random=100,**kwargs):    
    """
    Calculates the following metrics: local and global efficiency, and
        normalized clustering, path length and small worldness.
    Inputs:
        matrix (n x n): adjacency binarized matrix
        n_random (int): number of randomized graph.
    Returns:
        local_efficiency
        global_efficiency
        normalized clustering = real clustering / randomized clustering
        normalized path length= real path length / randomized path length
        small worldness = normalized clustering / normalized path length
    Notes
        F. Shi et al. (2012) Neuroimage 62 1622- 1633.
    """
    rewired_matrices=(generate_random_matrix(matrix) for _ in range(n_random))
    #create a graph using networkx.
    G=from_numpy_array(matrix)
    #efficiency
    global_eff=global_efficiency(G)
    local_eff=local_efficiency(G)
    try:
        if not kwargs['small_worldness']:
            return local_eff, global_eff
    except KeyError:
        pass
        
    #clustering and path length
    def clustering_coefficient_and_path_length(G):
        clustering_values=clustering(G)
        clustering_values=np.mean([abs(value) for value in clustering_values.values()])
        path_length=[]
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            path_length.append(average_shortest_path_length(C))
        path_length=np.mean(path_length)
        return clustering_values,path_length
    
    clustering_G, path_length_G = clustering_coefficient_and_path_length(G)
    
    clustering_rewired_G_total=[]
    path_length_rewired_G_total=[]
    for _ in range(n_random):
        rewired_G=from_numpy_array(next(rewired_matrices))
        clustering_rewired_G, path_length_rewired_G = clustering_coefficient_and_path_length(rewired_G)
        clustering_rewired_G_total.append(clustering_rewired_G)
        path_length_rewired_G_total.append(path_length_rewired_G)
    
    normalized_clustering = clustering_G/np.mean(clustering_rewired_G_total)
    normalized_path_length = path_length_G/np.mean(path_length_rewired_G_total)
    
    small_worldness = normalized_clustering/normalized_path_length

    return local_eff, global_eff, normalized_clustering, normalized_path_length, small_worldness

def preprocess_perm_graph_results(shape,**kwargs):
    """
    Results have been run simultaneously on several virtual machines, and
        sometimes the connection is broken. The results are written to a .txt
        file, and each permuted example has 30 lines corresponding to 30
        thresholds of network sparsity.
    Args:
        shape: (-1,30,5)
        high_risk (list of file paths)
        low_risk (list of file paths)
    Return:
        high_risk (array)
        low risk (array)
    """
    def new_processed_matrix(f):
        temp_result=np.loadtxt(f)
        n_sample=temp_result.shape[0]-(temp_result.shape[0]%30)
        return temp_result[0:n_sample,:]
    if len(kwargs['high_risk'])!=len(kwargs['low_risk']):
        return ('something wrong')
    perm_high=[]
    perm_low=[]
    for idx,i in enumerate(kwargs['high_risk']):
        try:
            #the files have vm and run number.
            if str.split(i,'vm')[1] not in kwargs['low_risk'][idx]:
                print('the following files do not correspond:',i,'and',kwargs['low_risk'][idx])
        except IndexError:
            pass
    
        temp_result_high=new_processed_matrix(kwargs['high_risk'][idx])
        temp_result_low=new_processed_matrix(kwargs['low_risk'][idx])
        if len(temp_result_low)!=len(temp_result_high):
            print('the two list do not have the same length:',i,'and',kwargs['low_risk'][idx])
            new_n_sample=len(min([temp_result_low,temp_result_high], key=len))
            temp_result_high=temp_result_high[0:new_n_sample,:]
            temp_result_low=temp_result_low[0:new_n_sample,:]

        perm_high.append(temp_result_high)
    perm_high=np.vstack(perm_high).reshape(shape)
    perm_low=np.vstack(perm_low).reshape(shape)
    return perm_high,perm_low

def calculate_perm_p_value(obsv_high,obsv_low,perm_high,perm_low,metrics=5):
    """
    Calculate the between group difference. and see if less than 5 percentile 
        of between-group difference in the permutation distribution was greater 
        than the observed group difference.
    Args:
        obsv_high and obsv_low (array of shape 30 x 5): observed high and low 
        risk graph metrics matrices
        perm_high and perm_low (array of shape n x 30 x 5):permuted high and 
        low graph metrics matrices.
    
    Returns:
        p_value_matrix (30 x metrics)
    """
    between_group_difference=abs(perm_high-perm_low)
    obsv_difference=abs(obsv_high-obsv_low)
    p_value_matrix=np.zeros((30,metrics))
    for network_sparsity_threshold in range(30): #30 thresholds
        for metric in range(metrics):# 5 metrics
            p_value_matrix[network_sparsity_threshold,metric]=len(np.where(between_group_difference[:,network_sparsity_threshold,metric]>=obsv_difference[network_sparsity_threshold,metric])[0])/between_group_difference.shape[0]
    return p_value_matrix


def calculate_path_and_efficiency_bin(G):
    """
    Return the following network metrics for your graph. Uses brain connectivity tool-box.
    Input:
        G (np.array): binarized matrix, symmetric
    Outputs:
        Eglob: Global efficiency
        average_Elocal: mean Local Efficiency
        char_path: Average shortest path length
        clustering: Average Clustering coefficient
    
    """
    if not np.allclose(G,G.T):
        raise bct.BCTParamError('Not undirected')

    average_shortest_path,Eglob,_,_,_=bct.charpath(bct.distance_bin(G))
    average_clustering=np.mean(abs(bct.clustering_coef_bu(G)))
    average_Elocal=np.mean(bct.efficiency_bin(G,1))

    return Eglob,average_Elocal,average_clustering,average_shortest_path

def calculate_network_metrics_bin_und(binarized_matrix,random_network_n=100):
    """
    Calculate the network metrics for the original and the randomized network
    """
    assert len(np.unique(binarized_matrix))==2#binarized matrix
    assert np.allclose(binarized_matrix,binarized_matrix.T)#undirected

    original_network_metrics = calculate_path_and_efficiency_bin(binarized_matrix)

    random_network=(bct.randmio_und(binarized_matrix,2)[0] for _ in range(random_network_n))

    random_network_metrics = np.zeros(4,)
    for _ in range(random_network_n):
        random_network_metrics += np.asarray(calculate_path_and_efficiency_bin(next(random_network))) #sum up all the randomizations.
    
    random_network_metrics = random_network_metrics/random_network_n #get the average values

    return np.array(original_network_metrics), random_network_metrics

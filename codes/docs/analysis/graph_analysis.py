import numpy as np
import pandas as pd
from scipy.linalg import lstsq as scipy_lstsq
import tqdm
from typing import List,Union,Optional
from collections import defaultdict

class Graph_analysis:
    
    @staticmethod
    def get_edge_size(X:np.ndarray, comps:np.ndarray, comp_sizes:np.ndarray):
        """
        Calculate for each connected component the edge size.
        The unconnected nodes are coded as an independent number with comp_sizes of 1.
        First we need to remove these unconnected nodes and retain only connected components.
        Then for each connected component, we find the nodes in that connected component.
        From there, we can look at the lower half of the binarised matrix, and count number of 1s in the matrix containing only those nodes.
        
        
        Parameters
        ----------
        X : np.ndarray
            the binarised squared symmetric matrix.
        comps : np.ndarray
            list of the size of number of nodes, where each location is the component the node belong to.
            e.g. [1,1,2,2] means the first two nodes belong to component 1, and the last 2 in component 2
        comp_sizes : np.ndarray
            Size of each component in terms of number of nodes.

        Returns
        -------
        ind_sz : np.ndarray
            This tells me which components id is connected [1,2] means component id 1, and 2 are connected.
        sz_links : np.ndarray
            This tells me for each component in ind_sz, how many edges there is.
        max_edge_size : float
            The max of sz_links. If sz_links = 0, then max_edge_size = 0.

        """
        #calculate the largest connected component size in terms of edges
        np.fill_diagonal(X,0) #this is needed because when I sum the square symmetric mesh, I don't want to count the diagonals.
        ind_sz, = np.where(comp_sizes>1) # see and removes the unconnected nodes
        ind_sz +=1 # by seeing which connected comps values correspond to ind_sz
        nr_components = np.size(ind_sz) # number of components
        sz_links = np.zeros((nr_components,))
        for i in range(nr_components):
            nodes, = np.where(ind_sz[i] == comps)
            sz_links[i] = np.sum(X[np.ix_(nodes,nodes)])/2 #get the number of edges for each connected components
        
        if np.size(sz_links):
            max_edge_size = np.max(sz_links)
        else:
            max_edge_size = 0
            
        return ind_sz, sz_links, max_edge_size
    
    @staticmethod
    def get_components(X:np.ndarray):
        """
        Returns the components of an undirected graph specified by the binary and
        undirected adjacency matrix adj. Components and their constitutent nodes
        are assigned the same index and stored in the vector, comps. The vector,
        comp_sizes, contains the number of nodes beloning to each component.
        Note:
            Taken directly from bct.clustering.get_components
            Cite accordingly
        Parameters
        ----------
        X : np.ndarray
            binarised matrix, the diagonal may not be 1.

        Raises
        ------
        ValueError
            If X is not undirected.

        Returns
        -------
        comps : np.ndarray
            list of the size of number of nodes, where each location is the component the node belong to.
            e.g. [1,1,2,2] means the first two nodes belong to component 1, and the last 2 in component 2
        comp_sizes : np.ndarray
            Size of each component in terms of number of nodes.
        max_edge_size: float
            
        Note:
            Unconnected nodes will have their own index component and component size of 1
            How do you calculate the size of a connected component?
            You obviously want to count the edges.
            But because unconnected nodes will have the size node of 1, you need to remove them
        """
        if not np.all(X == X.T):
            raise ValueError('get_components requires an undirected matrix')
        n = len(X)
        np.fill_diagonal(X, 1)
        edge_map = [{u,v} for u in range(n) for v in range(n) if X[u, v] == 1] # get the edges, where u is the row and v is the column indices
        union_sets = []
        for item in edge_map:
            # check for each edge in the edge map,
            # if the first edge has common node with the second edge, append them together.
            # else, add them as separate component
            temp = []
            for s in union_sets:
                if not s.isdisjoint(item):
                    item = s.union(item)
                else:
                    temp.append(s)
            temp.append(item)
            union_sets = temp
        comps = np.array([i+1 for v in range(n) for i in range(len(union_sets)) if v in union_sets[i]])# assign each component with a number
        comp_sizes = np.array([len(s) for s in union_sets]) # get the component size
        #calculate the largest connected component size in terms of edges
        _,_, max_edge_size = Graph_analysis.get_edge_size(X, comps, comp_sizes)
        
        return comps, comp_sizes, max_edge_size
    
    @staticmethod
    def get_degree_und(X:np.ndarray):
        #X is binarised matrix
        node_degree = np.sum(X,axis=0)
        return node_degree
    
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
    def reverse_lower_triangle(matrix,side_of_the_square=None):
        """
        Organise a 1D matrix to a square 2D undirected matrix.
        Args:
            matrix: 1D matrix
            side_of_the_square (int): desired square size 
        Returns:
            matrix: 2D matrix
        """
        if side_of_the_square is None:
            side_of_the_square = int(np.roots([1,-1,-(len(matrix)*2)])[0])
            
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
    def combine_nodes_together(X:np.ndarray,
                               *idx_args,
                               **operation_kwargs):
        """
        Group the nodes together by summing their values in matrix (or mean, max)
        Return the new matrix or new names of the updated list

        Parameters
        ----------
        X : np.ndarray
            Matrix undirected of size (nodes,nodes).
        *idx_args : List or arrays
            Series of list or arrays, where each list is a group of nodes separated by comma.
        **operation_kwargs : TYPE
            operation: {'sum','mean','max'}
            grouped_node_list List or dict: 
                if dict: key = name of the group, value = indices in the group, idx_args is not required
                if list: this is list of name that correspond to each group. Must match the number of list in idx_args
            
            original_node_list: Original list of name for each node

        Raises
        ------
        ValueError
            If X is not undirected.
            If number of grouped idx is less than 2
        KeyError
            If providing grouped node list without original node list

        Returns
        -------
        X_new: np.ndarray
            Updated grouped X
        Grouped node list: np.ndarray
            Grouped node list, the position of each value correspond to each node.
        """
        if not np.all(X == X.T):
            raise ValueError('combine_regions requires undirected matrix')
        new_X = X.copy()
        if 'operation' not in operation_kwargs: # define what you want to do with groupings
            operation_kwargs['operation'] = 'sum'
            
        if 'grouped_node_list' in operation_kwargs: # node list correspond to the grouped idx_args:
            if 'original_node_list' not in operation_kwargs:
                raise KeyError('original_node_list is required to provide updated list')
            original_list = operation_kwargs['original_node_list']    
            if isinstance(operation_kwargs['grouped_node_list'],dict):
                grouped_idx_name = operation_kwargs['grouped_node_list']
                idx_args = list(grouped_idx_name.values())
                grouped_idx_name = {idx:name for name,indices in grouped_idx_name.items() for idx in indices}
            elif isinstance(operation_kwargs['grouped_node_list'],list):
                #change the name in the original list to match with grouped node list and then drop duplicates keep first
                grouped_idx_name = {idx:grouped_name for idx_list,grouped_name in zip(idx_args, operation_kwargs['grouped_node_list']) for idx in idx_list }
            updated_node_list = [(idx,grouped_idx_name[idx]) if idx in grouped_idx_name.keys() else (idx,original_name)
                    for idx,original_name in zip(np.arange(X.shape[0]),original_list)]
            updated_node_list = pd.Series(dict(updated_node_list)).drop_duplicates(keep='first')
            updated_node_list = updated_node_list.to_numpy()
        
        for idx_list in idx_args:
            if len(idx_list)<2:
                continue
            rows_to_group = new_X[idx_list,:] # pick out the rows you want to group
            if operation_kwargs['operation'] =='sum':
                new_row = rows_to_group.sum(axis=0)
            elif operation_kwargs['operation'] == 'mean':
                new_row = rows_to_group.mean(axis=0)
            elif operation_kwargs['operation'] == 'max':
                new_row = rows_to_group.max(axis=0)
            new_X[idx_list[-1],:] = new_row # write out the group result to the last index row
            new_X[:,idx_list[-1]] = new_row # and the last index column
            new_X[idx_list[:-1],:] = 0 # set all other columns and rows to 0
            new_X[:,idx_list[:-1]] = 0
            np.fill_diagonal(new_X, 0)
        new_X = new_X[~(new_X==0).all(axis=1)] # remove rows with 0s
        new_X = new_X[:,~(new_X==0).all(axis=0)] # remove columns with 0s
        
        if 'grouped_node_list' in operation_kwargs:
            return new_X,updated_node_list
        else:
            return new_X
    
    @staticmethod
    def remove_nodes(X:np.ndarray,
                     nodes_to_remove:np.ndarray,
                     original_node_list:List[str]=None):
        """
        Removes the nodes from the matrix and return a new matrix and list of updated nodes name

        Parameters
        ----------
        X : np.ndarray
            Matrix undirected of size (nodes,nodes)..
        nodes_to_remove : np.ndarray
            Index of nodes in the matrix that needs to be removed.
        original_node_list : List[str], optional
            Name of the nodes in the matrix. The default is None.

        Raises
        ------
        ValueError
            If X is not undirected.

        Returns
        -------
        new_X: np.ndarray
            new updated matrix.
        updated_node_list : List
            list of updated node names

        """
        if not np.all(X==X.T):
            raise ValueError('requires undirected matrix')
        new_X = X.copy()
        np.fill_diagonal(new_X, 1)
        new_X[nodes_to_remove,:] = 0
        new_X[:,nodes_to_remove] = 0
        new_X = new_X[~(new_X==0).all(axis=1)] # remove rows with 0s
        new_X = new_X[:,~(new_X==0).all(axis=0)] # remove columns with 0s
        np.fill_diagonal(new_X, 0)
        if not original_node_list:
            return new_X
        updated_node_list = [i for idx,i in enumerate(original_node_list) if idx not in nodes_to_remove]
        return new_X, updated_node_list
    
        
            
class NBS:
    
    class T_matrix:
    
        @staticmethod
        def calculate_t_matrix(X:np.ndarray,y:np.ndarray):
            """
            Calculate t-value by performing mass linear regression.
            Utilises scipy.linalg.lstsq lapack driver = 'gelsy'
            Each linear regression model has a bias term, but t-value of only the predictor is outputed.
            Parameters
            ----------
            X : np.ndarray
                Matrix of shape (n,K) where n is the number of observation and K is the number of connections.
            y : np.ndarray
                The target continuous variable, which will be shuffled in permutation testing.
            Returns
            -------
            t_matrix : np.ndarray
                Assuming the undirected graph, this is a (K,) vector where t_matrix[0] correspond to the t-value of the K[0] in linear regression model with a bias term.
            """
            const = np.ones((X.shape[0],1))
            t_matrix = np.empty(X.shape[1])
            for dependentVar in range(X.shape[1]):
                X_new = np.append(const,X[:,dependentVar:dependentVar+1],axis=1)
                beta, _, _, _ = scipy_lstsq(X_new,y,lapack_driver='gelsy')
                y_pred = np.matmul(X_new,beta)
                MSE = (np.sum((y-y_pred)**2))/(len(X)-2)
                var_b = MSE*(np.linalg.inv(np.dot(X_new.T,X_new)).diagonal())
                sd_b = np.sqrt(var_b)
                ts_b = beta/sd_b
                t_matrix[dependentVar] = ts_b[1]
            return t_matrix
        
        @staticmethod
        def get_permutation(X:np.ndarray,y:np.ndarray, perm:int=1000)->np.ndarray:
            """
            Parameters
            ----------
            X : np.ndarray
                Matrix of shape (n,K) where n is the number of observation and K is the number of connections.
            y : np.ndarray
                The target continuous variable, which will be shuffled in permutation testing
            perm : int, optional
                number of permutation. The default is 1000. uses np.random.Generator.permutation (without replacement)
    
            Returns
            -------
            all_t_perm : np.ndarray
                A t-value matrix of shape (perm,K) where perm is the number of permutation performed, and K is the number of connection examined.
    
            """
            rng = np.random.default_rng(42)
            all_t_perm = np.empty((perm,X.shape[1]))
            
            for perm_n in tqdm.tqdm(range(perm)):
                y_perm = rng.permutation(y)
                t_matrix_perm = NBS.T_matrix.calculate_t_matrix(X, y_perm)
                all_t_perm[perm_n,:] = t_matrix_perm
                    
            return all_t_perm
    
    class Corr_matrix:
        @staticmethod
        def calculate_adj_corr_matrix(X:np.ndarray,y:np.ndarray,to_permute:np.ndarray=None):
        #calculate adjusted values for y and then perform x
            const = np.ones((X.shape[0],1))
            adjusted_y = np.empty(y.shape)
            if y.ndim == 1:
                raise ValueError('y cannot be less than 1 to calculate correlation matrix')
            if isinstance(to_permute,np.ndarray):
                if to_permute.ndim == 1:
                    to_permute = to_permute.reshape(-1,1)
                X = np.append(X,to_permute,axis=1)
            X_new = np.append(const,X,axis=1)
            for value in range(y.shape[1]):
                beta, _, _, _ = scipy_lstsq(X_new,y[:,value],lapack_driver='gelsy')
                y_pred = np.matmul(X_new,beta)
                adjusted_y[:,value] = y[:,value] - y_pred
            corr_matrix = np.corrcoef(adjusted_y,rowvar=False)
            corr_matrix = Graph_analysis.lower_triangle(corr_matrix)
            return corr_matrix
        
        @staticmethod
        def get_permutation_corr_matrix(X:np.ndarray,y:np.ndarray,to_permute:np.ndarray=None,perm=1000):
            rng = np.random.default_rng(42)
            all_corr_perm = np.empty((perm,int((y.shape[1])*(y.shape[1]-1)/2)))
            
            for perm_n in tqdm.tqdm(range(perm)):
                to_permute = rng.permutation(to_permute)
                corr_perm = NBS.Corr_matrix.calculate_adj_corr_matrix(X, y,to_permute=to_permute)
                all_corr_perm[perm_n,:] = corr_perm
                    
            return all_corr_perm

    
    @staticmethod
    def get_null_distribution(permutation_matrix:np.ndarray,
                              threshold:float=None,
                              number_of_nodes:int=None,
                              metrics:Optional[Union[List[str],str]]=None):
        """
        Applying a t-value threshold to the t-value matrix derived from permutation step.
        And calculate the maximal size of the connected component.

        Parameters
        ----------
        all_t_perm : np.ndarray
            A t-value matrix of shape (perm,K) where perm is the number of permutation performed, and K is the number of connection examined..
        threshold : float
            The threshold to be applied to all_t_perm.
        number_of_nodes : int, optional
            Number of nodes examined.
        metrics: dict
            {'max_edge_size','max_degree_centrality'}

        Returns
        -------
        null_distribution : np.ndarray
            Vector of shape (perm,), at each position is the maximal edge size for each permutation.
        """
        permutation_matrix = np.where(abs(permutation_matrix)>threshold,1,0) #apply supra-threshold
        null_distribution = defaultdict(np.ndarray)
        if isinstance(metrics,list):
            for metric in metrics:
                null_distribution[metric] = np.empty(permutation_matrix.shape[0])
        for row in range(permutation_matrix.shape[0]):
            temp_square = Graph_analysis.reverse_lower_triangle(permutation_matrix[row,:],side_of_the_square=number_of_nodes)
            if 'max_edge_size' in metrics:
                _, _, max_edge_size = Graph_analysis.get_components(temp_square)# get the component and component size
                null_distribution['max_edge_size'][row] = max_edge_size
            if 'max_degree_centrality' in metrics:
                null_distribution['max_degree_centrality'][row] = np.max(Graph_analysis.get_degree_und(temp_square)) # maximum degree centrality
                
        return null_distribution
            
    @staticmethod
    def get_p_val(observed_matrix,
                  threshold:float,
                  null_distribution:dict,
                  number_of_nodes:int=None):
        """
        Get p value for each connected component in the observed data

        Parameters
        ----------
        X : np.ndarray
            Matrix of shape (n,K) where n is the number of observation and K is the number of connections.
        y : np.ndarray
            The target continuous variable, which will be shuffled in permutation testing.
        threshold : float
            The t-value threshold to be applied to t-value matrix.
        null_distribution : np.ndarray
            Vector of shape (perm,), at each position is the maximal edge size for each permutation.
        number_of_nodes : int, optional
            Number of nodes examined. The default is 90.

        Returns
        -------
        p_vals : np.ndarray
            Matrix of shape (nr_connected_component,) corresponding to p-value at each connected component
        comps : np.ndarray
            list of the size of number of nodes, where each location is the component the node belong to.
            e.g. [1,1,2,2] means the first two nodes belong to component 1, and the last 2 in component 2
        ind_sz : np.ndarray
            This tells me which components id is connected [1,2] means component id 1, and 2 are connected.
        sz_links : np.ndarray
            This tells me for each component in ind_sz, how many edges there is.
        """
        if not isinstance(observed_matrix, np.ndarray):
            raise TypeError('matrix must be a numpy array of shape (k,)')
        observed_matrix = np.where(abs(observed_matrix)>threshold,1,0) #apply supra-threshold
        observed_square = Graph_analysis.reverse_lower_triangle(observed_matrix,side_of_the_square=number_of_nodes)
        observed_metrics = defaultdict(dict)
        for k in null_distribution:
            if k == 'max_edge_size':
                comps, comp_sizes,_ = Graph_analysis.get_components(observed_square)
                #calculate number of edges
                ind_sz, sz_links, _ = Graph_analysis.get_edge_size(observed_square, comps, comp_sizes)
                #calculate p_vals
                p_vals = np.zeros((ind_sz.shape))
                for i in range(len(ind_sz)):
                    p_vals[i] = np.size(np.where(null_distribution[k] >= sz_links[i])) / len(null_distribution[k])
                observed_metrics[k]['p_vals'] = p_vals
                observed_metrics[k]['comps'] = comps
                observed_metrics[k]['ind_sz'] = ind_sz
                observed_metrics[k]['sz_links'] = sz_links
            if k == 'max_degree_centrality':
                node_degree = Graph_analysis.get_degree_und(observed_square)
                p_vals = np.zeros((node_degree.shape))
                for i in range(len(node_degree)):
                    p_vals[i] = np.size(np.where(null_distribution[k] >= node_degree[i])) / len(null_distribution[k])
                observed_metrics[k]['p_vals'] = p_vals
                observed_metrics[k]['node_degree'] = node_degree
        
        return observed_metrics



import os
import numpy as np
import warnings
import networkx as nx
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
import Feasibility
import pickle as pickle
from scipy.spatial import distance

random_seed = 482

class GraphBuilder:
    def __init__(self, feasibility_constraints, feature_columns, X, kernel, exclude_columns=True):
        self.feasibility_constraints = feasibility_constraints
        self.feature_columns = feature_columns
        self.N = X.shape[0]
        self.kernel = kernel
        self.G = nx.DiGraph()
        self.edges = []
        self.densities = {}
        self.exclude_columns = exclude_columns
        self.temp_nodes = set()

    def compute_pairwise_distances_within_subgroups_and_graph(self, datasetName, data, epsilon, feasibility_constraints_instance, roundb):
        subgroups = self.group_data_based_on_constraints(datasetName, data)
        pairwise_distances = np.zeros((len(data), len(data)))
    
        for node in [node_index for node_index in range(pairwise_distances.shape[0])]:
            self.G.add_node(node)

        data_values = data.to_numpy()
        pairwise_distances = distance.cdist(data_values, data_values, 'euclidean')
        if roundb:
            pairwise_distances = np.round(pairwise_distances, roundb)

        for subgroup_key, subgroup in subgroups.items():
            if datasetName == 'GermanCredit':
                subgroup_distances = self.pairwise_distances_and_graph_german_credit(subgroup, epsilon, pairwise_distances, feasibility_constraints_instance)
            else:
                subgroup_distances = self.pairwise_distances_and_graph(subgroup, epsilon, pairwise_distances, feasibility_constraints_instance)
        return pairwise_distances, self.G, self.densities

    def group_data_based_on_constraints(self, datasetName, data):
        if datasetName == "Heloc":
            return {'no_group': data}
        subgroups = {}
        
        if datasetName == 'Student':
            for sex_value, sex_subgroup in data.groupby('sex'):
                subgroups[sex_value] = sex_subgroup
            
        if datasetName == 'GermanCredit':
            for sex_value, sex_subgroup in data.groupby('Sex'):
                subgroups[sex_value] = sex_subgroup

            self.feasibility_constraints.set_feature_columns_to_check(["Sex"])
            
        elif datasetName in ['Compas', 'Adult']:
            sex_column = 'sex'
            race_columns = [col for col in data.columns if col.startswith('race_')]            

            for sex_value in data[sex_column].unique():
                for race_value in range(len(race_columns)):
                    sex_race_subgroup = data[(data[sex_column] == sex_value) & (data[f'race_{race_value}'] == 1)]
                    subgroups[(sex_value, race_value)] = sex_race_subgroup

            
            if self.exclude_columns:
                self.excluded_columns = ['sex'] + race_columns
                self.feasibility_constraints.set_feature_columns_to_check(self.excluded_columns)
        return subgroups
    
    def pairwise_distances_and_graph(self, data, epsilon, pairwise_distances, feasibility_constraints_instance):
        data_values = data.to_numpy()
        edges = []

        for i in tqdm(range(len(data)), desc="Building graph for subgroup..."):
            for j in range(i + 1, len(data)):
                index_i = int(data.index[i])
                index_j = int(data.index[j])
                data_i = data_values[i]
                data_j = data_values[j]
                dist = None
                xy = False
                yx = False
                dist = pairwise_distances[index_i, index_j]

                if dist < epsilon:
                    xy = feasibility_constraints_instance.check_constraints(data_i, data_j)
                    yx = feasibility_constraints_instance.check_constraints(data_j, data_i)
                    if xy or yx:
                        density = self.kernel.kernelKDE(data_i, data_j, dist)
                        wij = dist * density
                        if isinstance(wij, np.ndarray) and wij.size == 1:
                            wij = wij.item()
                        if xy:
                            edges.append((index_i, index_j, {'distance': dist, 'wij': wij}))
                            self.densities[(index_i, index_j)] = density
                        if yx:
                            edges.append((index_j, index_i, {'distance': dist, 'wij': wij}))
                            self.densities[(index_j, index_i)] = density
        self.G.add_edges_from(edges)
    
    def pairwise_distances_and_graph_german_credit(self, data, epsilon, pairwise_distances, feasibility_constraints_instance):
        data_values = data.to_numpy()
        edges = []

        for i in tqdm(range(len(data)), desc="Building graph for subgroup..."):
            for j in range(i + 1, len(data)):
                index_i = int(data.index[i])
                index_j = int(data.index[j])
                data_i = data_values[i]
                data_j = data_values[j]
                dist = None
                xy = False
                yx = False
                dist = pairwise_distances[index_i, index_j]

                if dist < epsilon:
                    xy = feasibility_constraints_instance.check_constraints_german_credit(data_i, data_j)
                    yx = feasibility_constraints_instance.check_constraints_german_credit(data_j, data_i)
                    if xy or yx:
                        density = self.kernel.kernelKDE(data_i, data_j, dist)
                        wij = dist * density
                        if isinstance(wij, np.ndarray) and wij.size == 1:
                            wij = wij.item()
                        if xy:
                            edges.append((index_i, index_j, {'distance': dist, 'wij': wij}))
                            self.densities[(index_i, index_j)] = density
                        if yx:
                            edges.append((index_j, index_i, {'distance': dist, 'wij': wij}))
                            self.densities[(index_j, index_i)] = density
        self.G.add_edges_from(edges)
    
def get_FN_Negatives_Positives(data, clf, tp, attr_col_mapping, FEATURE_COLUMNS, TARGET_COLUMNS, index_mapping):
    """
    This function will return the False Negatives and all the negatively classified data points

    Parameters:
    -----------
    data: pandas DataFrame
    clf: classifier
    tp: float
        Threshold probability for classification
    attr_col_mapping: dict
        Mapping of column names to their corresponding indices in the data DataFrame
    FEATURE_COLUMNS: list of feature columns
    TARGET_COLUMNS: list of target columns

    Returns:
    --------
    FN: dict
        Dictionary containing the false negatives with DataFrame indexes as keys
    negative_points: dict
        Dictionary containing all negatively classified data points with DataFrame indexes as keys
    positive_points: dict
        Dictionary containing all positively classified data points with DataFrame indexes as keys
    """
    negative_points = {}  
    FN = {}
    positive_points = {}
    FP = {}

    for x_id, x in enumerate(data):
        original_index = index_mapping[x_id]
        features = x[[attr_col_mapping[col] for col in FEATURE_COLUMNS]]
        if clf.__class__.__name__ == "XGBClassifier":
            if clf.predict_proba([features])[0][1] > tp:
                positive_points[original_index] = x
                if x[attr_col_mapping[TARGET_COLUMNS]] == 0:
                    FP[original_index] = x                
            elif clf.predict([features]) == 0:
                negative_points[original_index] = x
                if x[attr_col_mapping[TARGET_COLUMNS]] == 1:
                    FN[original_index] = x
        elif clf.__class__.__name__ == "LogisticRegression":
            if clf.predict_log_proba([features])[0][1] > np.log(tp):
                positive_points[original_index] = x
                if data[x_id, attr_col_mapping[TARGET_COLUMNS]] == 0:
                    FP[original_index] = x
            elif clf.predict([features]) == 0:
                negative_points[original_index] = x
                if data[x_id, attr_col_mapping[TARGET_COLUMNS]] == 1:
                    FN[original_index] = x
        elif clf.__class__.__name__ == "Sequential":     
            x_features = x[[attr_col_mapping[col] for col in FEATURE_COLUMNS]].reshape(1, -1)
            prediction_probabilities = clf.predict(x_features)
            predicted_class = (prediction_probabilities > tp).astype(int)
            predicted_class = predicted_class[0, 0] 

            if predicted_class == 1:
                positive_points[original_index] = x
                if x[attr_col_mapping[TARGET_COLUMNS]] == 0:
                    FP[original_index] = x
            else:
                negative_points[original_index] = x
                if x[attr_col_mapping[TARGET_COLUMNS]] == 1:
                    FN[original_index] = x
    return FN, negative_points, positive_points, FP

def get_normalized_group_identifier_value(group_identifier, group_identifier_value, min_max_scaler, data_df_copy):
    """
    This function will normalize the group identifier value

    # Parameters:
    -----------
    - group_identifier: str
    - group_identifier_value: int
    - min_max_scaler: object
    - data_df_copy: pandas DataFrame

    # Returns:
    --------
    - normalized_group_identifier_value: float
    """
    min_value = min_max_scaler.data_min_[data_df_copy.columns.get_loc(group_identifier)]
    max_value = min_max_scaler.data_max_[data_df_copy.columns.get_loc(group_identifier)]

    normalized_value = (group_identifier_value - min_value) / (max_value - min_value)
    return normalized_value

def get_false_negatives_by_group(FN, group_identifier_column, group_identifier_value, data, numeric_columns):
    """
    Get the false negatives grouped by a column.

    # Parameters:
    ------------
    - FN: dict
        Dictionary containing the false negatives with DataFrame indexes as keys.
    - group_identifier_column: str
        The name of the column containing the group identifiers.
    - group_identifier_value: int
        The value of the group identifier.
    - data: pandas DataFrame
        The input data.
    - numeric_columns: list
        The list of numeric columns in the dataset.
    
    # Returns:
    -----------
    - groups: dict
        A dictionary where keys are group identifiers and values are corresponding false negatives.
    """
    groups = {}

    if group_identifier_column in numeric_columns:
        group_less = {}
        group_greater = {}

        for k, v in FN.items():
            if data.loc[k, group_identifier_column] < group_identifier_value:
                group_less[k] = v
            else:
                group_greater[k] = v
        
        groups[f"{group_identifier_value}_less"] = group_less
        groups[f"{group_identifier_value}_greater"] = group_greater
    
    else:
        unique_identifiers = np.unique(data[group_identifier_column])
        for identifier in unique_identifiers:
            groups[identifier] = {k: v for k, v in FN.items() if data.loc[k, group_identifier_column] == identifier}
    
    return groups

def getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name):
    """
    Get the feasibility constraints for a dataset.

    # Parameters:
    ------------
    - FEATURE_COLUMNS: list
        The list of feature columns in the dataset.
    - dataset_name: str
        The name of the dataset.

    # Returns:
    -----------
    - feasibility_constraints_instance: object
        An instance of the Feasibility class.
    """
    feasibility_constraints_instance = Feasibility.feasibility_consts(FEATURE_COLUMNS)
    if (dataset_name == 'german_credit'):
        feasibility_constraints_instance.set_constraint('No of dependents', mutability=False)
        feasibility_constraints_instance.set_constraint('Sex & Marital Status', mutability=False)               
        feasibility_constraints_instance.set_constraint('Age (years)', step_direction=1)
        feasibility_constraints_instance.set_constraint('Duration in Current address', step_direction=1)
        feasibility_constraints_instance.set_constraint('Length of current employment', step_direction=1)
    elif (dataset_name == 'Student'):
        feasibility_constraints_instance.set_constraint('sex', mutability=False)
        feasibility_constraints_instance.set_constraint('age', step_direction=1)
        feasibility_constraints_instance.set_constraint('famsize', step_direction=1)
        feasibility_constraints_instance.set_constraint('Medu', step_direction=1)
        feasibility_constraints_instance.set_constraint('Fedu', step_direction=1)
        feasibility_constraints_instance.set_constraint('nursery', step_direction=1)
        feasibility_constraints_instance.set_constraint('health', step_direction=1)
    elif (dataset_name == 'Compas'):
        feasibility_constraints_instance.set_constraint('race', mutability=False, exact_match=True)
        feasibility_constraints_instance.set_constraint('sex', mutability=False)
        feasibility_constraints_instance.set_constraint('age', step_direction=1)
        feasibility_constraints_instance.set_constraint('priors_count', step_direction=-1)
        feasibility_constraints_instance.set_constraint('juv_fel_count', step_direction=-1)
        feasibility_constraints_instance.set_constraint('juv_misd_count', step_direction=-1)
        feasibility_constraints_instance.set_constraint('juv_other_count', step_direction=-1)
        feasibility_constraints_instance.set_constraint('c_charge_degree', step_direction=-1)                                                                                                                                   
    elif (dataset_name == 'Heloc'):                                                                                                                                                                                                                                                               
        feasibility_constraints_instance.set_constraint('NumSatisfactoryTrades', step_direction=1)                                                                                                              
        feasibility_constraints_instance.set_constraint('PercentTradesNeverDelq', step_direction=1)
        feasibility_constraints_instance.set_constraint('MSinceMostRecentDelq', step_direction=-1)
        feasibility_constraints_instance.set_constraint('NetFractionInstallBurden', step_direction=-1)
        feasibility_constraints_instance.set_constraint('MaxDelqEver', step_direction=-1)
        feasibility_constraints_instance.set_constraint('MaxDelq2PublicRecLast12M', step_direction=-1)
        feasibility_constraints_instance.set_constraint('NumTrades60Ever2DerogPubRec', step_direction=-1)
        feasibility_constraints_instance.set_constraint('NumTrades90Ever2DerogPubRec', step_direction=-1)
        feasibility_constraints_instance.set_constraint('MaxDelq2PublicRecLast12M', step_direction=-1)
        feasibility_constraints_instance.set_constraint('ExternalRiskEstimate', step_direction=-1)
    elif (dataset_name == 'Adult'):
        feasibility_constraints_instance.set_constraint('age', step_direction=1)
        feasibility_constraints_instance.set_constraint('education', step_direction=1, exact_match=False)
        feasibility_constraints_instance.set_constraint('educational-num', step_direction=1)
        feasibility_constraints_instance.set_constraint('race', mutability=False, exact_match=True)
        feasibility_constraints_instance.set_constraint('sex', mutability=False)
    else:
        print("Unknown dataset. Initializing with no constraints.")
    return feasibility_constraints_instance

def get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier_column, group_identifier_value, numeric_columns):
    """
    Get the subgraphs of a graph based on the group identifiers.

    Parameters:
        graph (nx.Graph): The input graph.
        data_np (np.ndarray): The numpy array representation of the dataframe.
        attr_col_mapping (dict): A dictionary mapping attribute names to column indices.
        group_identifier_column (str): The name of the column containing the group identifiers.

    Returns:
        dict: A dictionary where keys are group identifiers and values are corresponding subgraphs.
    """
    subgraphs = {}
    col_index = attr_col_mapping[group_identifier_column]

    if group_identifier_column in numeric_columns:
        identifier_less = f"{group_identifier_value}_less"
        identifier_greater = f"{group_identifier_value}_greater"
        subgraphs[identifier_less] = graph.subgraph([node for node in graph.nodes() if data.loc[node, group_identifier_column] <= group_identifier_value])
        subgraphs[identifier_greater] = graph.subgraph([node for node in graph.nodes() if data.loc[node, group_identifier_column] > group_identifier_value])
    else:
        unique_identifiers = np.unique(data_np[:, col_index])
        for identifier in unique_identifiers:
            subgraph_nodes = [node for node, attrs in graph.nodes(data=True) if data.loc[node, group_identifier_column] == identifier]
            subgraphs[identifier] = graph.subgraph(subgraph_nodes)
    return subgraphs
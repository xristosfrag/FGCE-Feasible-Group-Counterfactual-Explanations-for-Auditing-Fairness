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

    def compute_distances_in_blocks(self, data, block_size=1000, representation=16):
        """Computes distances in smaller blocks and rounds results to save memory.
        
        Understanding Memory Usage:
        Supposing a dataset of 40,000 samples.
        The pairwise distance matrix is 40,000 x 40,000.
        Since storing distances in a float64 (8 bytes per value), the total memory required is:
        40,000 x 40,000 x 8 bytes = 12.8 GB (approximately).
        
        Expected Memory Savings
            Data Type	Memory per Value	Full Matrix Size
            float64	    8 bytes	            12.8 GB
            float32	    4 bytes	            6.4 GB
            float16	    2 bytes	            3.2 GB
        """
        n = len(data)
        if representation == 16:
            decimals = np.float16
        elif representation == 32:
            decimals = np.float32
        elif representation == 64:
            decimals = np.float64
        
        results = np.zeros((n, n), dtype=decimals)
        for i in range(0, n, block_size):
            for j in range(i, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                results[i:i_end, j:j_end] = distance.cdist(data[i:i_end], data[j:j_end], 'euclidean').astype(decimals)
                if i != j:
                    results[j:j_end, i:i_end] = results[i:i_end, j:j_end].T  # Symmetric assignment
        return results
    
    def compute_pairwise_distances_within_subgroups_and_graph(self, datasetName, data, epsilon, feasibility_constraints_instance, representation):
        subgroups = self.group_data_based_on_constraints(datasetName, data)
        pairwise_distances = np.zeros((len(data), len(data)))
    
        for node in [node_index for node_index in range(pairwise_distances.shape[0])]:
            self.G.add_node(node)

        data_values = data.to_numpy()
        if representation:
            pairwise_distances = self.compute_distances_in_blocks(data_values, block_size=1000, representation=representation)
        else:
            pairwise_distances = distance.cdist(data_values, data_values, 'euclidean')

        for subgroup_key, subgroup in subgroups.items():
            if datasetName == 'GermanCredit':
                self.pairwise_distances_and_graph_german_credit(subgroup, epsilon, pairwise_distances, feasibility_constraints_instance)
            else:
                self.pairwise_distances_and_graph(subgroup, subgroup_key, epsilon, pairwise_distances, feasibility_constraints_instance)
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
            
        elif datasetName in ['Compas', 'Adult', 'AdultLouisiana', 'AdultCalifornia']:
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

    def pairwise_distances_and_graph(self, data, subgroup_key, epsilon, pairwise_distances, feasibility_constraints_instance):
        data_values = data.to_numpy()
        index_list = data.index.to_numpy()
        len_data = len(data)
        edges = []
        densities = {}

        print("Building graph for subgroup {} with {} nodes...".format(subgroup_key, len_data))
        for i in range(len_data):
            index_i = index_list[i]
            data_i = data_values[i]
            for j in range(i + 1, len_data):
                index_j = index_list[j]
                data_j = data_values[j]
                dist = pairwise_distances[index_i, index_j]
                if dist > epsilon: 
                    continue
                    
                mean_data = (data_i + data_j) * 0.5
                if feasibility_constraints_instance.check_constraints(data_i, data_j):
                    densities[(index_i, index_j)] = mean_data
                if feasibility_constraints_instance.check_constraints(data_j, data_i):
                    densities[(index_j, index_i)] = mean_data
        if densities != {}:
            self.densities.update(self.kernel.kernelKDEdata(densities, pairwise_distances))
        
        for (idx_i, idx_j), density in self.densities.items():
            dist = pairwise_distances[idx_i, idx_j]
            edges.append((idx_i, idx_j, {'distance': dist, 'wij': density * dist}))

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

def serialize_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {str(k): serialize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return [serialize_json(i) for i in obj]
    return obj

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
    if (dataset_name == 'GermanCredit'):
        feasibility_constraints_instance.set_constraint('Month-Duration', step_direction=-1)
        feasibility_constraints_instance.set_constraint('Credit-Amount', step_direction=-1)      
        feasibility_constraints_instance.set_constraint('Instalment-Rate', step_direction=-1)
        feasibility_constraints_instance.set_constraint('Installment', step_direction=-1) 
        feasibility_constraints_instance.set_constraint('Property', step_direction=-1)
        feasibility_constraints_instance.set_constraint('Existing-Account-Status', step_direction=1)
        feasibility_constraints_instance.set_constraint('Savings-Account', step_direction=1)
        feasibility_constraints_instance.set_constraint('Age', step_direction=1)
        feasibility_constraints_instance.set_constraint('Present-Employment', step_direction=1)
        feasibility_constraints_instance.set_constraint('Guarantors', step_direction=1)
        feasibility_constraints_instance.set_constraint('Housing', step_direction=1)
        feasibility_constraints_instance.set_constraint('Job', step_direction=1)
        feasibility_constraints_instance.set_constraint('Telephone', step_direction=1)
        feasibility_constraints_instance.set_constraint('Sex', mutability=False)
        feasibility_constraints_instance.set_constraint('Marital-Status', mutability=False, exact_match=True)
        feasibility_constraints_instance.set_constraint('Foreign-Worker', mutability=False)
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
        feasibility_constraints_instance.set_constraint('education', step_direction=1)
        feasibility_constraints_instance.set_constraint('educational-num', step_direction=1)
        feasibility_constraints_instance.set_constraint('race', mutability=False, exact_match=True)
        feasibility_constraints_instance.set_constraint('sex', mutability=False)
    elif (dataset_name == "AdultLouisiana" or dataset_name == "AdultCalifornia"):
        feasibility_constraints_instance.set_constraint('age', step_direction=1)
        feasibility_constraints_instance.set_constraint('sex', mutability=False)
        feasibility_constraints_instance.set_constraint('race', mutability=False, exact_match=True)
        feasibility_constraints_instance.set_constraint('Educational Attainment', step_direction=1)
        feasibility_constraints_instance.set_constraint('Place of Birth', mutability=False)
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

# =====================================================================================================================
# =====================================================================================================================
#                 		 					ATTRIBUTION
# =====================================================================================================================
# =====================================================================================================================
dataset_feature_descriptions = {
    "Student": {
        "sex": "Student's sex",
        "age": "Student's age",
        "Medu": "Mother's education level",
        "Fedu": "Father's education level",
        "Mjob": "Mother's job",
        "Fjob": "Father's job",
        "reason": "Reason to choose this school",
        "guardian": "Student's guardian",
        "traveltime": "Travel time to school",
        "studytime": "Weekly study time",
        "failures": "Num. of past class failures",
        "schoolsup": "Extra educational support",
        "famsup": "Family educational support",
        "paid": "Extra paid classes within the course subject",
        'activities': "Extra-curricular activities",
        "nursery": "Attended nursery school",
        "higher": "Wants to take higher education",
        "internet": "Access to internet at home",
        "romantic": "In a romantic relationship",
        "famrel": "Quality of family relationships",
        "freetime": "Free time after school",
        "goout": "Going out with friends",
        'Dalc': 'Workday alcohol consumption',
        'Walc': 'Weekend alcohol consumption',
        "health": "Current health status",
        "absences": "Num. of school absences",
        "Pstatus": "Parent's cohabitation status"
    },
    "Compas": {
        "sex": "Defendant's sex",
        "age": "Defendant's age",
        "juv_fel_count": "Num. of juvenile felonies",
        "juv_misd_count": "Num. of juvenile misdemeanors",
        "juv_other_count": "Num. of other juvenile offenses",
        "priors_count": "Num. of prior charges",
        "age_cat": "Defendant's age category",
        "race": "Defendant's race",
        "c_charge_degree": "Degree of the charge",
        "c_charge_desc": "Description of the charge"
    },
    "Heloc": {
        "AverageMInFile": "Average months in file for all tradelines",
        "NetFractionInstallBurden": "Net fraction of installment credit to credit limit",
        "NetFractionRevolvingBurden": "Net fraction of revolving credit to credit limit",
        "PercentInstallTrades": "Installment trades (%)",
        "PercentTradesWBalance": "Trades with balance (%)",
        "NumTotalTrades": "Total number of credit trades",
        "MSinceMostRecentDelq": "Months since most recent delinquency",
        "NumSatisfactoryTrades": "Num. of satisfactory credit trades",
        "PercentTradesNeverDelq": "Trades that have never been delinquent (%)",

        "NumTradesOpeninLast12M": "Num. of trades opened in last 12 months",
        "MSinceMostRecentTradeOpen": "Months since most recent trade open",

        "ExternalRiskEstimate": "Risk estimate from external source",
        "MSinceOldestTradeOpen": "Months since oldest trade open",
        "NumTrades60Ever2DerogPubRec": "Num. of trades 60+ ever 2 derogatory public records",
        "NumTrades90Ever2DerogPubRec": "Num. of trades 90+ ever 2 derogatory public records",
        "MaxDelq2PublicRecLast12M": "Max delinquency in 12 months",
        "MaxDelqEver": "Max delinquency ever",
        "NumTotalTradesOpenLast12M": "Num. of total trades open in last 12 months",
        "MSinceMostRecentInqexcl7days": "Months since most recent inquiry excluding 7 days",
        "NumInqLast6M": "Num. of inquiries in last 6 months",
        "NumInqLast6Mexcl7days": "Num. of inquiries in last 6 months excluding 7 days",

        "NumRevolvingTradesWBalance": "Num. of revolving trades with balance",
        "NumInstallTradesWBalance": "Num. of installment trades with balance",
        "NumBank2NatlTradesWHighUtilization": "Num. of bank/national trades with high utilization",
    },
    # "Adult": {
    #     "age": "Age",
    #     "workclass": "Employment status",
    #     "sex": "Sex",
    #     "capital-gain": "Capital gain",
    #     "capital-loss": "Capital loss",
    #     "hours-per-week": "Hours worked per week",
    #     "education": "Education level",
    #     "education-num": "Years of Education",
    #     "marital-status": "Marital status",
    #     "occupation": "Occupation",
    #     "relationship": "Relationship status"
    # },
    "Adult": {
        "age": "Age",
        "workclass": "Employment",
        "sex": "Sex",
        "capital-gain": "Cap. gain",
        "capital-loss": "Cap. loss",
        "hours-per-week": "Work Hrs",
        "education": "Education",
        "education-num": "Edu. Yrs",
        "marital-status": "Marital",
        "occupation": "Occupation",
        "relationship": "Relationship"
    },
    "AdultCalifornia": {
        "age": "Age",
        "Class of Worker": "Employment status",
        "sex": "Sex",
        "Educational Attainment": "Education level",
        "Marital Status": "Marital status",
        "Occupation": "Occupation",
        "Place of Birth": "Place of birth",
        "Hours Worked per Week": "Hours worked per week",
    },
    "AdultLouisiana": {
        "age": "Age",
        "Class of Worker": "Employment status",
        "sex": "Sex",
        "Educational Attainment": "Education level",
        "Marital Status": "Marital status",
        "Occupation": "Occupation",
        "Place of Birth": "Place of birth",
        "Hours Worked per Week": "Hours worked per week",
    },
    "GermanCredit":{
        "CreditAmount": "Amount of credit required",
        'Existing-Account-Status': 'Balance or type of the checking account',
        'Month-Duration': 'Credit duration in months',
        'Savings-Account': 'Savings account/bonds',
        'Present-Employment': 'Duration of present employment',
        'Instalment-Rate': 'Installment rate in percentage of disposable income',
        'Sex': 'Sex of applicant',
        'Guarantors': 'Presence of guarantors',
        'Residence': 'Duration in present residence',
        'Property': 'Property ownership',
        'Age': 'Age of applicant',
        'Installment': 'Other installment plans',
        'Housing': 'Housing situation',
        'Existing-Credits': 'Number of existing credits at this bank',
        'Job': 'Job status',
        'Num-People': 'Number of people being liable to provide maintenance for',
        'Telephone': 'Registered telephone',
        'Foreign-Worker': 'Whether the applicant is a foreign worker',
        'Credit-History': 'Past credit behaviour of individual',
        'Purpose': 'Purpose of credit',
        'Marital-Status': "Marital status"
    }
}
dataset_one_hot_mapping = {
    "Student": {
        "Mjob": {
            "0": "teacher",
            "1": "health",
            "2": "services",
            "3": "at_home",
            "4": "other"
        },
        "Fjob": {
            "0": "teacher",
            "1": "health",
            "2": "services",
            "3": "at_home",
            "4": "other"
        },
        "guardian": {
            "0": "mother",
            "1": "father",
            "2": "other"
        },
        "reason": {
            '0': 'course',
            '1': 'other',
            '2': 'home',
            '3': 'reputation'
        }
    },
    "Compas": {
        "race": {
            "0": "Other",
            "1": "African-American",
            "2": "Caucasian",
            "3": "Hispanic",
            "4": "Asian",
            "5": "Native American"
        },
        "age_cat": {
            "0": "Less than 25",
            "1": "25-45",
            "2": "Greater than 45"
        }
    },
    "Adult":{
        "race": {
            "0": "White",
            "1": "Asian-Pac-Islander",
            "2": "Amer-Indian-Eskimo",
            "3": "Other",
            "4": "Black"
        }
    },
    "AdultCalifornia": {
        "race":{
            '0': 'White', 
            '1': 'Black or African American', 
            '2': 'American Indian', 
            '3': 'Alaska Native', 
            '4': 'American Indian or Alaska Native (tribes specified or not)', 
            '5': 'Asian', 
            '6': 'Native Hawaiian and Other Pacific Islander', 
            '7': 'Some Other Race', 
            '8': 'Two or More Races'
        }
    },
    "AdultLouisiana": {
        "race":{
            '0': 'White', 
            '1': 'Black or African American', 
            '2': 'American Indian', 
            '3': 'Alaska Native', 
            '4': 'American Indian or Alaska Native (tribes specified or not)', 
            '5': 'Asian', 
            '6': 'Native Hawaiian and Other Pacific Islander', 
            '7': 'Some Other Race', 
            '8': 'Two or More Races'
        }
    },
    "GermanCredit": {
        'Credit-History': {
            '0': 'no credits taken/ all credits paid back duly',
            '1': 'all credits at this bank paid back duly',
            '2': 'existing credits paid back duly till now',
            '3': 'delayed in paying off in the past',
            '4': 'critical account/ other credits existing (not at this bank)',
        },
        'Purpose': {
            '0': 'car (new)',
            '1': 'car (used)',
            '2': 'furniture/equipment',
            '3': 'radio/television',
            '4': 'domestic appliances',
            '5': 'repairs',
            '6': 'education',
            '7': 'retraining',
            '8': 'business',
            '9': 'others',
        },
        'Marital-Status': {
            '0': 'single',
            '1': 'married',
            '2': 'divorced',
            '3': 'other'
        }
    }
}
import json
import sys
import numpy as np
import pickle as pk
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from xgboost import XGBClassifier

import utils
from utils import GraphBuilder
from FGCE import *
from kernel import *
from dataLoader import *


def get_FGCE_Directory():
    """Get the path of the 'FGCE-Feasible-Group-Counterfactual-Explanations-for-Auditing-Fairness' directory."""
    current_dir = os.getcwd()
    target_dir = 'FGCE-Feasible-Group-Counterfactual-Explanations-for-Auditing-Fairness'
    
    while os.path.basename(current_dir) != target_dir:
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            return None
        
    return current_dir

def get_path_separator():
    """Get the system-specific directory separator."""
    return os.sep

FGCE_DIR = get_FGCE_Directory()
sys.path.append(FGCE_DIR)
sep = get_path_separator()

def initialize_FGCE(epsilon=3, tp=0.6, td=0.001, datasetName='Student', 
					group_identifier='sex', classifier="lr", bandwith_approch="mean_scotts_rule", group_identifier_value=None, 
					skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True, skip_bandwith_calculation=True, roundb=None):
	data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, continuous_featues, one_hot_encode_features = load_dataset(datasetName=datasetName)
	"""
	Initialize the FGCE algorithm
	
	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	-tp: (float)
		Prediction threshold
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FGCE algorithm
	- bandwith_approch: (str)
		the method to use for calculating the bandwith
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
		load model or not
	- skip_distance_calculation: (boolean)
		load distances or not
	- skip_graph_creation: (boolean)	
		load graph or not
	- skip_bandwith_calculation: (boolean)
		load bandwith or not
	- roundb: (int)
		the number of decimal places to round calculated distances
	
	# Returns:
	----------------
	- fgce: (FGCE object)
		the initialized FGCE object
	- graph: (networkx graph)
		the graph object
	- distances: (dict)
		the pairwise distances
	- data: (pandas DataFrame)
		the data
	- data_np: (numpy array)
		the data points
	- data_df_copy: (pandas DataFrame)
		the copy of the original data
	- attr_col_mapping: (dict)
		the mapping of the column names to their indices
	- normalized_group_identifer_value: (float)
		the normalized value of the group identifier
	- numeric_columns: (list)
		the list of numeric columns
	- positive_points: (list)
		the list of positive points
	- FN: (list)
		the list of false negatives
	- FN_negatives_by_group: (dict)
		the false negatives by group
	- node_connectivity: (float)
		the node connectivity
	- edge_connectivity: (float)
		the edge connectivity
	"""
	if 'GermanCredit' in datasetName:
		datasetName = 'GermanCredit'
	print("Data shape:", data.shape)
	TEST_SIZE = 0.3
	X = data[FEATURE_COLUMNS]

	data_np = data.to_numpy()
	attr_col_mapping = {col: i for i, col in enumerate(data.columns)}
	X = data_np[:, [attr_col_mapping[col] for col in FEATURE_COLUMNS]]

	X_train, X_test, y_train, y_test = train_test_split(
		data[FEATURE_COLUMNS],
		data[TARGET_COLUMNS],
		test_size=TEST_SIZE,
		random_state=utils.random_seed,
		shuffle=True
	)

	print("Data size:", X_train.shape)
	print("Data Train columns:", data.columns)
	print("Target columns:", TARGET_COLUMNS)

	if not os.path.exists(f"{FGCE_DIR}{sep}tmp"):
		os.makedirs(os.path.join(FGCE_DIR, 'tmp'))

	if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}"):
		os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}")
		
	train_model = None
	param_grid = None
	model = None
	if classifier == "lr":
		if skip_model_training and "LR_classifier_data.pk" in os.listdir(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			print("Loading classifier from file ...")
			clf = pk.load(open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}LR_classifier_data.pk", "rb"))
		else:
			clf = LogisticRegression(random_state=utils.random_seed)
			clf.fit(X_train, y_train)

			print("Training accuracy:", clf.score(X_train, y_train))
			print("Testing accuracy:", clf.score(X_test, y_test))

			if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}"):
				os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}")

			pk.dump(clf, open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}LR_classifier_data.pk", 'wb'))

	elif classifier == "xgb":
		if skip_model_training and "XGB_classifier_data.pk" in os.listdir(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			print("Loading classifier from file ...")
			model = pk.load(open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}XGB_classifier_data.pk", "rb"))
		else:
			param_grid = {
			'n_estimators': [50, 100, 200, 500],  # Increase upper bound
			'max_depth': [3, 5, 7, 10, 15],  # Add deeper trees
			'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Improve learning rate choices
			'subsample': [0.5, 0.7, 0.9, 1],  # Add 0.9 to test near-full dataset
			'colsample_bytree': [0.5, 0.7, 0.9, 1],  # Add 0.9 for better diversity
			'gamma': [0, 0.1, 0.5, 1, 5],  # Prevent overfitting
			'reg_alpha': [0, 0.01, 0.1, 1],  # L1 regularization
			'reg_lambda': [1, 5, 10],  # L2 regularization
			}

			model = xgb.XGBClassifier(
				objective="binary:logistic",
				eval_metric="logloss"
			)
			train_model = 'xgb'
	elif classifier == "rf":
		if skip_model_training and "RF_classifier_data.pk" in os.listdir(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			print("Loading classifier from file ...")
			model = pk.load(open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}RF_classifier_data.pk", "rb"))
		else:
			param_grid = {
			'n_estimators': [100, 200, 300, 400],  # Number of trees
			'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
			'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
			'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be a leaf node
			'bootstrap': [True, False],  # Whether to use bootstrap sampling
			}
			model = RandomForestClassifier(random_state=42)
			train_model = 'rf'
	elif classifier == "dnn":
		if skip_model_training and "DNN_classifier_data.h5" in os.listdir(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}"):
			print("Loading classifier from file ...")
			model = tf.keras.models.load_model(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}DNN_classifier_data.keras")
		else:
			def create_dnn_model(optimizer='adam', dropout_rate=0.5, hidden_units=32):
				model = Sequential()
				model.add(Input(shape=(X_train.shape[1],)))
				model.add(Dense(hidden_units, activation='relu'))
				model.add(Dropout(dropout_rate))
				model.add(Dense(hidden_units // 2, activation='relu'))
				model.add(Dense(1, activation='sigmoid'))
				model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
				return model

			# Wrap the model using KerasClassifier
			model = KerasClassifier(
				model=create_dnn_model,
				verbose=0,
				epochs=10,  # Lower epochs for faster tuning
				batch_size=32
			)

			# Define hyperparameter search space
			param_grid = {
				'model__optimizer': ['adam', 'rmsprop'],
				'model__dropout_rate': [0.3, 0.5],
				'model__hidden_units': [32, 64],
				'batch_size': [8, 16],  # Smaller batch sizes
				'epochs': [5, 10]
			}
			train_model = 'dnn'
	else:
		raise ValueError("Invalid classifier type. Supported types are 'lr', 'xgb', and 'dnn'.")

	if train_model != None:	
		# Perform hyperparameter tuning
		random_search = RandomizedSearchCV(
			estimator=model,
			param_distributions=param_grid,
			n_iter=15,  # Run more iterations for better search
			cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
			scoring='balanced_accuracy',  # Better than recall-only
			## dont assign all cpus to the search. istead assing max - 5
			n_jobs=max(1, (os.cpu_count() or 1) - 5),
			verbose=0,
			random_state=42
		)

		print(f"Starting {classifier} hyperparameter search...")
		random_search.fit(X_train, y_train)

		# Retrieve best model (already trained during hyperparameter search)
		model = random_search.best_estimator_

		# Print results
		print(f"\nBest {classifier} Hyperparameters: {random_search.best_params_}")
		print(f"Best cross-validated accuracy: {random_search.best_score_:.4f}")
		print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
		print(f"Testing Accuracy: {model.score(X_test, y_test):.4f}")

	if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}"):
		os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}")
	if train_model == 'lr':
		pk.dump(model, open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}LR_classifier_data.pk", 'wb'))
	elif train_model == 'xgb':
		pk.dump(model, open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}XGB_classifier_data.pk", 'wb'))
	elif train_model == 'rf':
		pk.dump(model, open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}RF_classifier_data.pk", 'wb'))
	elif train_model == 'dnn':
		model.save(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}DNN_classifier_data.keras")


	start_time = time.time()

	FN, negative_points, positive_points, FP = utils.get_FN_Negatives_Positives(data_test, clf, tp, attr_col_mapping, FEATURE_COLUMNS, TARGET_COLUMNS, index_mapping)

	print(f"FN: {len(FN)} - Negatives: {len(negative_points)} - FP: {len(FP)} - Positives: {len(positive_points)}")
	normalized_group_identifer_value = None
	if group_identifier in numeric_columns and group_identifier_value is not None:
		print(f"Group identifier value: {group_identifier_value}")
		normalized_group_identifer_value = utils.get_normalized_group_identifier_value(group_identifier, group_identifier_value, min_max_scaler, data_df_copy)
		print(f"Normalized group identifier value: {normalized_group_identifer_value}")
	elif group_identifier in numeric_columns and group_identifier_value is None:
		raise ValueError(f"The group_identifier column {group_identifier} does not contain numerical values")

	FN_negatives_by_group = utils.get_false_negatives_by_group(FN, group_identifier, normalized_group_identifer_value, data, numeric_columns)

	if skip_distance_calculation and skip_graph_creation and\
				os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}Graph_{epsilon}.pkl")\
					and os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Distances.pkl") and os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Densities{sep}Densities_{epsilon}.pkl"):
		print("Loading graph from file ...")
		graph = pk.load(open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}Graph_{epsilon}.pkl", "rb"))
		kernel = Kernel(datasetName, X, skip_bandwith_calculation=skip_bandwith_calculation, bandwith_approch=bandwith_approch)
		kernel.fitKernel(X)
		fgce = FGCE(data_np, X, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, clf)
		feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)
		fgce.set_graph(graph)

		print("Loading distances from file ...")
		distances = pk.load(open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Distances.pkl", "rb"))
	else:
		kernel = Kernel(datasetName, X, skip_bandwith_calculation=skip_bandwith_calculation, bandwith_approch=bandwith_approch)
		kernel.fitKernel(X)
		fgce = FGCE(data_np, X, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, clf)
		feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)

		start_time = time.time()
		dng_obj = GraphBuilder(feasibility_constraints, FEATURE_COLUMNS, X, kernel, exclude_columns=True)
		distances, graph, densities = dng_obj.compute_pairwise_distances_within_subgroups_and_graph(datasetName, data[FEATURE_COLUMNS], epsilon, feasibility_constraints, roundb)
		end_time = time.time()
		execution_time = end_time - start_time
		print("Distances and graph initialization: ", execution_time, " seconds")

		if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}"):
			os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}")
		if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Densities{sep}"):
			os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Densities")

		pk.dump(graph, open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Graphs{sep}Graph_{epsilon}.pkl", "wb"))
		pk.dump(distances, open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Distances.pkl", "wb"))
		pk.dump(densities, open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}Densities{sep}Densities_{epsilon}.pkl", "wb"))

	print(f"Max distance in the dataset: {np.sqrt(len(FEATURE_COLUMNS))}")
	print(f"Max possible distance considered in graph: {np.max(distances)}")
	
	fully_connected_nodes = len(X)
	connected_nodes = len(graph)
	node_connectivity = connected_nodes / fully_connected_nodes * 100
	print(f"{connected_nodes} nodes are connected out of {fully_connected_nodes} nodes. Connectivity: {node_connectivity}%")

	fully_connected_edges = (fully_connected_nodes * (fully_connected_nodes - 1)) / 2
	connected_edges = len(graph.edges())
	edge_connectivity = connected_edges /fully_connected_edges * 100
	print(f"{connected_edges} edges are connected out of {fully_connected_edges} edges. Connectivity: {edge_connectivity}%")

	end_time = time.time()
	execution_time = end_time - start_time
	print("FGCE initialization: ", execution_time, " seconds")

	return fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points, FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints


# =====================================================================================================================
# =====================================================================================================================
#                 		 					cost-constrained group counterfactuals
# =====================================================================================================================
# =====================================================================================================================
def main_cost_constrained_GCFEs(epsilon=3, tp=0.6, td=0.001, datasetName='Student', 
					group_identifier='sex', classifier="lr", bandwith_approch="mean_scotts_rule",
					k=5, max_d = 1, cost_function = "max_vector_distance", k_selection_method="greedy_accross_all_ccs", 
					group_identifier_value=None, skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True, 
					skip_bandwith_calculation=True,  skip_gcfe_calculation=True, compare_with_Face=False, roundb=None):
	"""
	This function is used to solve the cost-constrained group counterfactuals problem using the greedy coverage algorithm

	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	-tp: (float)
		Prediction threshold
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FGCE algorithm
	- k: (int)
		maximum number of cfes to return for each group
	- max_d: (int)
		maximum path cost for reachability (d-reachable threshold)
	- cost_function: (str)
		the cost function to use for the FGCE algorithm
	- k_selection_method: (str)
		the method to use for selecting the k cfes
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
		if it will skip the model training and load it if it exists or not
	- skip_distance_calculation: (boolean)
		if it will skip the distance calculation and load it if it exists or not
	- skip_graph_creation: (boolean)
		if it will skip the graph creation and load it if it exists or not
	- skip_bandwith_calculation: (boolean)
		if it will skip the bandwith calculation and load it if it exists or not
	- skip_gcfe_calculation: (boolean)
		if it will skip the group cfes calculation and load it if it exists or not
	- compare_with_Face: (boolean)
		if it will compare the results with the Face algorithm or not
	- roundb: (int)
		the number of decimal places to round calculated distances
	

	# Returns:
	----------------
	- results: (dict)
		dictionary containing the final results of the FGCE-Group algorithm
	- data_np: (numpy array)
		the data points
	- attr_col_mapping: (dict)
		the mapping of the column names to their indices
	- data_df_copy: (pandas DataFrame)
		the copy of the original data
	In case of comparing with the Face algorithm, the following values are returned:
	- face_vector_distances: (float)
		the average vector distance of the Face algorithm
	- gfce_vector_distances: (float)
		the average vector distance of the FGCE algorithm
	- face_wij_distances: (float)
		the average path cost of the Face algorithm
	- gfce_wij_distances: (float)
		the average path cost of the FGCE algorithm
	"""
	fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points, FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon, tp, td, datasetName, group_identifier, classifier, bandwith_approch, group_identifier_value, 
				 skip_model_training, skip_distance_calculation, skip_graph_creation, skip_bandwith_calculation, roundb)
	# =========================================================================================================================
	# 												GROUP CFES
	# =========================================================================================================================
	results = {}

	if cost_function == "max_vector_distance":
		max_d_store = round(max_d, 2)	
	elif cost_function == "max_path_cost":
		max_d_store = f"{max_d:.2e}"
	else:
		max_d_store = max_d
		
	if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs"):
		os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs")

	file_path = f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}cost_constrained_GCFEs{sep}results_{datasetName}_eps{epsilon}_tp{tp}_k_{k}_cost_function_{cost_function}_d_{max_d_store}_kmethod_{k_selection_method}.json"
	
	if skip_gcfe_calculation and os.path.exists(file_path):
		results = json.load(open(file_path, "r"))
		return results, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0
	else:
		start_time = time.time()
		subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
		print(f"\n{len(subgroups)} subgroups created based on group identifier: {group_identifier}")		
		
		print(f"Computing group cfes...")
		gcfes, not_possible_to_cover_fns_group = fgce.compute_gcfes_greedy(subgroups, positive_points, FN, max_d, cost_function, k, distances, k_selection_method)

		stats = {}
		stats["Node Connectivity"] = node_connectivity
		stats["Edge Connectivity"] = edge_connectivity

		results = fgce.apply_cfes(gcfes, FN_negatives_by_group, distances, not_possible_to_cover_fns_group, k_selection_method, cost_function, stats)

		end_time = time.time()
		execution_time = end_time - start_time
		print("Group Cfes - Time:", execution_time, "seconds")

		if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}cost_constrained_GCFEs"):
			os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}cost_constrained_GCFEs")

		with open(file_path, "w") as outfile:
			json.dump(results, outfile)
    # =========================================================================================================================
    # 											COMPARE WITH INDIVIDUAL CFES FROM FACE
    # =========================================================================================================================
	if compare_with_Face:
		fgce.set_candidates(positive_points)

		face_vector_distances = 0
		face_wij_distances = 0
		face_max_wij_distance = 0
		face_max_vector_distance = 0

		gfce_vector_distances = 0
		gfce_wij_distances = 0
		gfce_max_wij_distance = 0
		gfce_max_vector_distance = 0
		fn_points_found = 0

		## create individual recourses for each Fn point for each group
		if k_selection_method == "greedy_accross_all_ccs":
			for key in results:
				if key in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats"]: continue

				for fn_point in results[key]:
					if fn_point in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]: continue
					
					fn_points_found += 1
					shortest_paths_info, min_target_id = fgce.compute_recourse(graph, int(fn_point), feasibility_constraints, True)
					
					print("* FN Point:", fn_point)

					if shortest_paths_info is not None and min_target_id is not None and shortest_paths_info != {}:
						print("	Face Info: ")
						shortest_path_info = shortest_paths_info[min_target_id]
						
						print(f"		CFE: {min_target_id}")
						print(f"		Path cost: {shortest_path_info['path_cost']}")
						print(f"		Path cost dist: {shortest_path_info['path_cost_dist']}")
						print(f"		Vector Distance: {distances[int(fn_point)][min_target_id]}")
						fvd = distances[int(fn_point)][min_target_id]
						face_vector_distances += fvd	

						if fvd > face_max_vector_distance:
							face_max_vector_distance = fvd
						fwd = shortest_path_info['path_cost']

						if fwd > face_max_wij_distance:
							face_max_wij_distance = fwd
						face_wij_distances += fwd
					
					print("	FGCE Info:")
					print(f"		CFE: {results[key][fn_point]['CFE_name']}")
					print(f"		Shortest Path cost: {results[key][fn_point]['Shortest_path_cost']}")
					print(f"		Shortest Path cost dist: {results[key][fn_point]['Shortest_paths_distance_cost']}")
					print(f"		Vector Distance: {results[key][fn_point]['Vector_distance']}")
					gfvd = results[key][fn_point]['Vector_distance']
					gfce_vector_distances += gfvd

					if gfvd > gfce_max_vector_distance:
						gfce_max_vector_distance = gfvd

					gfwd = results[key][fn_point]['Shortest_path_cost']
					if gfwd > gfce_max_wij_distance:
						gfce_max_wij_distance = gfwd
					gfce_wij_distances += gfwd

		if fn_points_found != 0:
			face_wij_distances = (face_wij_distances / face_max_wij_distance) / fn_points_found
			gfce_wij_distances = (gfce_wij_distances / gfce_max_wij_distance) /fn_points_found

			face_vector_distances = (face_vector_distances / face_max_vector_distance)  / fn_points_found
			gfce_vector_distances = (gfce_vector_distances / gfce_max_vector_distance) / fn_points_found
		else:
			return results, data_np, attr_col_mapping, data_df_copy, None, None, 0, 0

		return results, data_np, attr_col_mapping, data_df_copy, face_vector_distances, gfce_vector_distances, face_wij_distances, gfce_wij_distances
	return results, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0


# =====================================================================================================================
# =====================================================================================================================
#                 		 					coverage-constrained group counterfactuals
# =====================================================================================================================
# =====================================================================================================================
def main_coverage_constrained_GCFEs(epsilon=0.2, tp=0.6, td=0.001, datasetName='Student', group_identifier='sex',
					classifier="lr", cost_function="max_path_cost", k=2,
					min_d=0, max_d=2, bst=1e-3, bandwith_approch="mean_scotts_rule",
					group_identifier_value=None, skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True,
					compare_with_Face=False, skip_gcfe_calculation=False,  skip_bandwith_calculation=True, find_k0=True, roundb=None):
	"""
	This function is used to solve the coverage-constrained group counterfactuals problem using binary search

	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	-tp: (float)
		Prediction threshold
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FGCE algorithm
	- k: (int)
		maximum number of cfes to return for each group
	- cost_function: (str)
		the cost function to use for the FGCE algorithm
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
		if it will skip the model training and load it if it exists or not
	- skip_distance_calculation: (boolean)
		if it will skip the distance calculation and load it if it exists or not
	- skip_graph_creation: (boolean)
		if it will skip the graph creation and load it if it exists or not
	- skip_bandwith_calculation: (boolean)
		if it will skip the bandwith calculation and load it if it exists or not
	- skip_gcfe_calculation: (boolean)
		if it will skip the group cfes calculation and load it if it exists or not
	- compare_with_Face: (boolean)
		if it will compare the results with the Face algorithm or not
	- find_k0: (boolean)
		if it will find the optimal k0 or not
	- roundb: (int)
		the number of decimal places to round calculated distances

	# Returns:
	----------------
	- results: (dict)
		dictionary containing the final results of the FGCE-Group algorithm
	- data_np: (numpy array)
		the data points
	- attr_col_mapping: (dict)
		the mapping of the column names to their indices
	- data_df_copy: (pandas DataFrame)
		the copy of the original data
	In case of comparing with the Face algorithm, the following values are returned:
	- face_vector_distances: (float)
		the average vector distance of the Face algorithm
	- gfce_vector_distances: (float)
		the average vector distance of the FGCE algorithm
	- face_wij_distances: (float)
		the average path cost of the Face algorithm
	- gfce_wij_distances: (float)
		the average path cost of the FGCE algorithm
	"""
	fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points, FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon, tp, td, datasetName, group_identifier, classifier, bandwith_approch, group_identifier_value, 
				 skip_model_training, skip_distance_calculation, skip_graph_creation, skip_bandwith_calculation, roundb)
	# =========================================================================================================================
	# 												GROUP CFES
	# =========================================================================================================================
	results = {}

	if cost_function == "max_vector_distance":
		max_d_store = round(max_d, 2)
		min_d_store = round(min_d, 2)
		bst_store = round(bst, 2)
	elif cost_function == "max_path_cost":
		max_d_store = f"{max_d:.2e}"
		min_d_store = f"{min_d:.2e}"
		bst_store = f"{bst:.2e}"
	else:
		max_d_store = max_d
		min_d_store = min_d
		bst_store = bst
	
	if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs"):
		os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs")

	file_path = f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs{sep}results_{datasetName}_eps{epsilon}_tp{tp}_k_{k}_cost_function_{cost_function}_max_d_store_{max_d_store}_min_d_store_{min_d_store}_bst_{bst_store}.json"
	
	if skip_gcfe_calculation and os.path.exists(file_path):
		results = json.load(open(file_path, "r"))
		return results, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0

	else:
		start_time = time.time()
		print(f"Computing group cfes...")

		subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
		print(f"\n{len(subgroups)} subgroups created based on group identifier: {group_identifier}")

		gcfes, gcfes_with_ccs, not_possible_to_cover_fns_group = fgce.compute_gcfes_binary(subgroups, positive_points, FN, k, min_d, max_d, bst, cost_function, distances, find_k0)

		stats = {}
		stats["Node Connectivity"] = node_connectivity
		stats["Edge Connectivity"] = edge_connectivity
		stats['Optimal d'] = {}

		for group in gcfes_with_ccs:
			d0s = []
			for cc in gcfes_with_ccs[group]:
				d0s.append(gcfes_with_ccs[group][cc]['optimal_d'])
			stats['Optimal d'][group] = np.max(d0s)
		
		for group in stats['Optimal d']:
			print(f"Group: {group} - Optimal d0: {stats['Optimal d'][group]}")
		
		stats = fgce.apply_cfes(gcfes, FN_negatives_by_group, distances, not_possible_to_cover_fns_group, "same_k_for_all_ccs", cost_function, stats, binary_implementation=True)

		end_time = time.time()
		execution_time = end_time - start_time
		print("Group Cfes - Time:", execution_time, "seconds")

		if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs"):
			os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs")

		with open(file_path, "w") as outfile:
			json.dump(stats, outfile)
	# =========================================================================================================================
    # 											COMPARE WITH INDIVIDUAL CFES FROM FACE
    # =========================================================================================================================
	if compare_with_Face:
		fgce.set_candidates(positive_points)

		face_vector_distances = 0
		face_wij_distances = 0
		face_max_wij_distance = 0
		face_max_vector_distance = 0

		gfce_vector_distances = 0
		gfce_wij_distances = 0
		gfce_max_wij_distance = 0
		gfce_max_vector_distance = 0
		fn_points_found = 0

		## create individual recourses for each Fn point for each group
		for key in stats:
			if isinstance(key, str):
				if key in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats"] or "Optimal d" in key:
					continue

			for fn_point in stats[key]:
				if fn_point in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]: continue

				for subkey in stats[key][fn_point]:
					if subkey != "CFE_name": continue
					cfe_name = stats[key][fn_point][subkey]

					if fn_point in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost", "Median path cost"]: continue
					fn_points_found += 1
					shortest_paths_info, min_target_id = fgce.compute_recourse(graph, int(fn_point), feasibility_constraints, True)
					
					print("* FN Point:", fn_point)

					if shortest_paths_info is not None and min_target_id is not None and shortest_paths_info != {}:
						print("	Face Info: ")
						shortest_path_info = shortest_paths_info[min_target_id]
						
						print(f"		CFE: {min_target_id}")
						fwd = shortest_path_info['path_cost']
						print(f"		Path cost: {fwd}")
						print(f"		Path cost dist: {shortest_path_info['path_cost_dist']}")
						fvd = distances[int(fn_point)][min_target_id]
						print(f"		Vector Distance: {fvd}")
						
						face_vector_distances += fvd	

						if fvd > face_max_vector_distance:
							face_max_vector_distance = fvd

						if fwd > face_max_wij_distance:
							face_max_wij_distance = fwd
						face_wij_distances += fwd
					
					print("	Face-Group Info:")
					print(f"		CFE: {cfe_name}")
					gfwd = stats[key][fn_point]['Shortest_path_cost']
					print(f"		Shortest Path cost: {gfwd}")
					print(f"		Shortest Path cost dist: {stats[key][fn_point]['Shortest_paths_distance_cost']}")
					gfvd = stats[key][fn_point]['Vector_distance']
					print(f"		Vector Distance: {gfvd}")
					
					gfce_vector_distances += gfvd

					if gfvd > gfce_max_vector_distance:
						gfce_max_vector_distance = gfvd

					if gfwd > gfce_max_wij_distance:
						gfce_max_wij_distance = gfwd
					gfce_wij_distances += gfwd

		if fn_points_found != 0:
			face_wij_distances = (face_wij_distances /face_max_wij_distance) /fn_points_found
			gfce_wij_distances = (gfce_wij_distances /gfce_max_wij_distance) /fn_points_found

			face_vector_distances = (face_vector_distances /face_max_vector_distance)   /fn_points_found
			gfce_vector_distances = (gfce_vector_distances /gfce_max_vector_distance) /fn_points_found
		else:
			return stats, data_np, attr_col_mapping, data_df_copy, None, None, 0, 0
		return stats, data_np, attr_col_mapping, data_df_copy, face_vector_distances, gfce_vector_distances, face_wij_distances, gfce_wij_distances
	return stats, data_np, attr_col_mapping, data_df_copy, [], [], 0, 0



# =====================================================================================================================
#                 		 					coverage-constrained group counterfactuals-MIP
# =====================================================================================================================

def main_coverage_constrained_GCFEs_MIP(epsilon=3, tp=0.6, td=0.001, datasetName='Student', 
					group_identifier='sex', classifier="lr", bandwith_approch="mean_scotts_rule", k=5, cost_function = "max_vector_distance",
					group_identifier_value=None, skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True,
					skip_gcfe_calculation=False,  skip_bandwith_calculation=True, cov_constr_approach="local", cov = 1,  roundb=None):
	"""
	This function is used to solve the coverage-constrained group counterfactuals problem using binary search

	# Parameters:
	----------------
	- epsilon: (float)
		margin for creating connections in graphs
	-tp: (float)
		Prediction threshold
	- datasetName: (str)
		name of the dataset
	- group_identifier: (str)	
		the column name of the group identifier
	- classifier: (str)
		the classifier to use for the FGCE algorithm
	- k: (int)
		maximum number of cfes to return for each group
	- cost_function: (str)
		the cost function to use for the FGCE algorithm
	- group_identifier_value: (int)
		the value of the group identifier
	- skip_model_training: (boolean)
	- skip_distance_calculation: (boolean)
		if it will skip the distance calculation and load it if it exists or not
	- skip_graph_creation: (boolean)
		if it will skip the graph creation and load it if it exists or not
	- skip_bandwith_calculation: (boolean)
		if it will skip the bandwith calculation and load it if it exists or not
	- skip_gcfe_calculation: (boolean)
		if it will skip the group cfes calculation and load it if it exists or not
	- cov_constr_approach: (str)
		the approach to use for the coverage constraint
	- roundb: (int)
		the number of decimal places to round calculated distances
	# Returns:
	----------------
	- results: (dict)
		dictionary containing the final results of the FGCE-Group algorithm
	"""
	fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points, FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon, tp, td, datasetName, group_identifier, classifier, bandwith_approch, group_identifier_value, 
				 skip_model_training, skip_distance_calculation, skip_graph_creation, skip_bandwith_calculation, roundb)
	# =========================================================================================================================
	# 												GROUP CFES
	# =========================================================================================================================
	results = {}	
	file_path = f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs{sep}{cov_constr_approach}{sep}stats{sep}results_{datasetName}_eps{epsilon}_tp{tp}_k_{k}_cov_{cov}_cost_function_{cost_function}.json"
	gcfes_path = f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs{sep}{cov_constr_approach}{sep}gcfes{sep}gcfes_{datasetName}_eps{epsilon}_tp{tp}_k_{k}_cov_{cov}_cost_function_{cost_function}.json"
	
	if skip_gcfe_calculation and os.path.exists(file_path):
		results = json.load(open(file_path, "r"))
		return results
	else:
		start_time = time.time()
		subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
		print(f"\n{len(subgroups)} subgroups created based on group identifier: {group_identifier}")		
		print(f"Computing group cfes...")

		if cov_constr_approach == "local":
			gcfes, results, _, _, _, _ = fgce.get_gcfes_approach_integer_prog_local(subgroups, distances, positive_points, FN) 
			
		elif cov_constr_approach == "global":
			gcfes, results = fgce.get_gcfes_approach_integer_prog_global(subgroups, distances, positive_points, FN, k, cov)
		
		end_time = time.time()
		if not os.path.exists(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs"):
			os.makedirs(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}coverage_constrained_GCFEs")

		with open(file_path, "w") as outfile:
			json.dump(results, outfile)
		with open(gcfes_path, "w") as outfile:
			json.dump(gcfes, outfile)
	return results
import sys
import os
import utils
import pickle as pk
import matplotlib.pyplot as plt

from FGCE import *
from main import *
from kernel import *
from dataLoader import *

import xgboost as xgb
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


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


def initialize_FGCE_attributes(datasetName='Student', skip_bandwith_calculation=True, bandwith_approch='optimal', classifier='xgb', skip_model_training=False):
    data, FEATURE_COLUMNS, TARGET_COLUMNS, _, _, \
        _, _, _, _ = load_dataset(datasetName=datasetName)
    if 'GermanCredit' in datasetName:
        datasetName = 'GermanCredit'
    X = data[FEATURE_COLUMNS]
    TEST_SIZE = 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMNS],
        test_size=TEST_SIZE,
        random_state=utils.random_seed,
        shuffle=True
    )

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
            model = pk.load(open(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}LR_classifier_data.pk", "rb"))
        else:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 200], 
                'solver': ['newton-cg', 'lbfgs', 'liblinear']
            }
            model = LogisticRegression(max_iter=10000)
            train_model = 'lr'
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
        model.model.save(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}DNN_classifier_data.keras")

    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
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
    positive_points = data[model.predict(data[FEATURE_COLUMNS]) == 1]
    print(f"Positive points: {len(positive_points)}")
    negative_points = X_test[model.predict(X_test[FEATURE_COLUMNS]) == 0]
    common_indices = negative_points.index.intersection(y_test[y_test == 1].index)
    FN = negative_points.loc[common_indices]
    print(f"FN: {len(FN)}")

    kernel = Kernel(datasetName, X, skip_bandwith_calculation=skip_bandwith_calculation, bandwith_approch=bandwith_approch)
    kernel.fitKernel(X)

    return data, data_np, X, FEATURE_COLUMNS, TARGET_COLUMNS, kernel, model

def plot(datasetName, face_dists, gfce_dists, face_wij, gfce_wij, d_method, max_d):
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    fig, ax1 = plt.subplots()

    # Adjust the x values slightly to avoid overlap
    x_values = range(1, len(face_wij)+1)
    x_values_offset = [x + 0.1 for x in x_values]
    x_values_offset2 = [x + 0.2 for x in x_values]

    ax1.plot(x_values, face_wij, '-o', color='green', label="Face Wij Distance", alpha=0.7)
    ax1.plot(x_values_offset, gfce_wij, '-o', color='red', label="FGCE Wij Distance", alpha=0.7)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Avg Wij Distance")
    ax1.set_xticks(range(1, len(face_wij)+1))

    ax2 = ax1.twinx()

    ax2.plot(x_values, face_dists, '-o', color='blue', label="Face Vector Distances", alpha=0.7)
    ax2.plot(x_values_offset2, gfce_dists, '-o', color='purple', label="FGCE Vector Distances", alpha=0.7)
    ax2.set_ylabel("Avg Vector Distance")

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    ax1.legend().remove()
    plt.tight_layout()
    fig_size = (6, 4) 
    plt.gcf().set_size_inches(fig_size)
    plt.savefig(f"{FGCE_DIR}/tmp/{datasetName}/figs/Coverage_constrained_face_gface_comparison_d_method_{d_method}_maxd_{max_d}_normalized.pdf")

    # Create a separate figure for the legend
    fig_legend = plt.figure(figsize=(1, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', fontsize=16)
    ax_legend.axis('off')
    fig_legend.tight_layout()
    fig_legend.savefig(f"{FGCE_DIR}/tmp/{datasetName}/figs/{datasetName}_legend.pdf")
    plt.show()
    
def nice_numbers(range_min, range_max, num_ticks, score='k', min_d=0.1):
    """
    Calculate "nice" numbers for the given range and number of ticks.

    Parameters
    ----------
    range_min : float
        The minimum value of the range.
    range_max : float
        The maximum value of the range.
    num_ticks : int
        The number of ticks to generate.
    score : str
        The score type ('k' or 'd'). Default is 'k'.

    Returns
    -------
    ticks : numpy.ndarray
        An array of "nice" numbers
    """
    # Calculate the range
    range_size = range_max - range_min

    # Calculate the approximate tick spacing
    tick_spacing = range_size / (num_ticks - 1)

    # Find a "nice" number for the tick spacing
    exponent = np.floor(np.log10(tick_spacing))
    fraction = tick_spacing / 10**exponent

    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3:
        nice_fraction = 2
    elif fraction < 7:
        nice_fraction = 5
    else:
        nice_fraction = 10

    nice_tick_spacing = nice_fraction * 10**exponent

    # Calculate the start and end points
    min_tick = np.ceil(range_min / nice_tick_spacing) * nice_tick_spacing
    max_tick = np.ceil(range_max / nice_tick_spacing) * nice_tick_spacing

    # Generate the ticks
    ticks = np.arange(min_tick, max_tick + nice_tick_spacing, nice_tick_spacing)

    if score == 'k':
        while len(ticks) != num_ticks:
            ticks = nice_numbers(range_min, range_max+1, num_ticks)
    elif score == 'd' and min_d>0.1:
        while len(ticks) != num_ticks:
            ticks = nice_numbers(range_min-0.1, range_max, num_ticks)    
    elif score == 'd':
        while len(ticks) != num_ticks:
            ticks = nice_numbers(range_min, range_max+0.1, num_ticks)

    return ticks

def face_comparison(datasetName="Student", epsilon=3, tp=0.5, td=0.001, bandwith_approch="mean_scotts_rule", classifier="xgb",\
                    group_identifier='sex', upper_limit_for_k=10, steps=10, group_identifier_value=None,\
                    skip_model_training=True, skip_bandwith_calculation=True, skip_graph_creation=True, skip_distance_calculation=True,\
                    max_d=1000000000, representation=64, bst=0.1):
    face_dists = []
    face_wij = []
    gfce_dists = []
    gfce_wij = []

    fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
                FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon=epsilon, tp=tp, td=td,\
                    datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch,\
                    group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_bandwith_calculation=skip_bandwith_calculation,\
                    skip_graph_creation=skip_graph_creation, skip_distance_calculation=skip_distance_calculation, representation=representation)
    fgce_init_dict = {"fgce": fgce, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
            "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
            "positive_points": positive_points, "FN": FN, "FN_negatives_by_group": FN_negatives_by_group, "node_connectivity": node_connectivity,\
                "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}

    k_values = nice_numbers(1, upper_limit_for_k, steps, score='k')
    for i, cfes in enumerate(k_values):
        print(f"Running for {i}-th time")

        results, data_np, attr_col_mapping, data_df_copy, face_vector_distances, gfce_vector_distances,\
        face_wij_distances, gfce_wij_distances = main_coverage_constrained_GCFEs(epsilon=epsilon, tp=tp, td=td,
                                datasetName=datasetName, group_identifier=group_identifier,
                                classifier='xgb', compare_with_Face=True, skip_distance_calculation=skip_distance_calculation,
                                skip_model_training=skip_model_training, skip_graph_creation=skip_graph_creation, skip_fgce_calculation=False,
                                k=cfes, max_d = max_d, cost_function="max_path_cost", fgce_init_dict=fgce_init_dict, bst=bst)

        if face_vector_distances == None:
            continue
        face_dists.append(face_vector_distances)
        gfce_dists.append(gfce_vector_distances)
        face_wij.append(face_wij_distances)
        gfce_wij.append(gfce_wij_distances)
    return face_dists, gfce_dists, face_wij, gfce_wij

def get_graph_stats(epsilon=0.4,\
        datasetName='Adult', group_identifier='sex', group_identifier_value=None, bandwith_approch="mean_scotts_rule", classifier='xgb',\
		skip_model_training=True, skip_distance_calculation=True, skip_graph_creation=True, skip_bandwith_calculation=True, verbose=False):
  
  fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
    FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints = initialize_FGCE(epsilon=epsilon,\
    datasetName=datasetName, group_identifier=group_identifier, bandwith_approch=bandwith_approch, classifier=classifier,\
    group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
    skip_graph_creation=skip_graph_creation, skip_bandwith_calculation=skip_bandwith_calculation, verbose=verbose)
      
  subgroups = utils.get_subgraphs_by_group(graph, data_np, data, attr_col_mapping, group_identifier, normalized_group_identifer_value, numeric_columns)
  weakly_connected_components = {}
  subgroup_nodes = {}
  stats = {}
  for group, subgraph in subgroups.items():
    weakly_connected_components[group] = list(nx.weakly_connected_components(subgraph))
    subgroup_nodes[group] = list(subgraph.nodes())

    strongly_connected_components = list(nx.strongly_connected_components(subgraph))
    
    density = nx.density(subgraph) 
    stats[group] = {'num_nodes': len(subgroup_nodes[group]), 'num_strongly_connected_components': len(strongly_connected_components),
            'num_weakly_connected_components': len(weakly_connected_components[group]), 'density': f'{density*100:.2f}'}
  return stats
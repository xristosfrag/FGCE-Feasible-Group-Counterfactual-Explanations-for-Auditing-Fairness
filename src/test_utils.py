import json
import sys
import numpy as np
import pickle as pk
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf

import utils
from utils import GraphBuilder
from FGCE import *
from kernel import *
from dataLoader import *
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


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
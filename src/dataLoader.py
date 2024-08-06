import os
import numpy as np
import pandas as pd
import pickle as pk
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from aif360.sklearn.datasets import fetch_compas

def get_FGCE_Directory():
    """Get the path of the 'FGCE-Feasible-Group-Counterfactual-Explanations-for-Auditing-Fairness' directory."""
    current_dir = os.getcwd()
    while os.path.basename(current_dir) != 'FGCE-Feasible-Group-Counterfactual-Explanations-for-Auditing-Fairness':
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            return None
        
    return current_dir
FGCE_DIR = get_FGCE_Directory()


def load_dataset(datasetName='Student'):
    if (datasetName == "Student"):
        return load_student()
    if (datasetName == "Adult"):
        return load_adult()
    if (datasetName == "Compas"):
        return load_compas()
    if (datasetName == "Heloc"):
        return load_heloc()
    if (datasetName == "GermanCredit"):
        return load_german_credit()

    return pd.DataFrame([]), [], [], None, None, None, None, [], []


def load_german_credit():
    data_df = pd.read_csv(data_df = pd.read_csv(f"{FGCE_DIR}/data/GermanCredit.csv"))
    TARGET_COLUMNS = data_df.columns[-1]
    data = data_df.drop(columns=[TARGET_COLUMNS])
    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)
    FEATURE_COLUMNS = data.columns
    data[TARGET_COLUMNS] = data_df[TARGET_COLUMNS]
    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, []

def load_student():
    data_df = pd.read_csv(f"{FGCE_DIR}/data/student.csv")
    TARGET_COLUMNS = data_df.columns[-1]
    data = data_df.drop(columns=[TARGET_COLUMNS])
    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])
    data_df_copy = data.copy()
    FEATURE_COLUMNS = data.columns
    data[TARGET_COLUMNS] = data_df[TARGET_COLUMNS]
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)
    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, []

def load_compas():
    X, y = fetch_compas()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    TARGET_COLUMNS = 'two_year_recid'
    data = X
    data = data.drop(['c_charge_desc', 'age_cat'], axis=1)
    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])
    data_df_copy = data.copy()
    y = pd.DataFrame(y, columns=[TARGET_COLUMNS])
    y, _, _ = preprocess_dataset(y, continuous_features=[])
    data[TARGET_COLUMNS] = y
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)
    FEATURE_COLUMNS = data.columns[:-1]
    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, []

def load_adult():
    data_df = pd.read_csv(f"{FGCE_DIR}/data/adult1.csv")
    TARGET_COLUMNS = data_df.columns[-1]
    data = data_df.drop(columns=[TARGET_COLUMNS])

    data['race'] = data['race'].astype(str)

    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])
    data_df_copy = data.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)
        
    FEATURE_COLUMNS = data.columns
    data[TARGET_COLUMNS] = data_df[TARGET_COLUMNS]

    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, []

def load_heloc():
    data_df = pd.read_csv(f"{FGCE_DIR}/data/heloc.csv")
    data_df = data_df[(data_df.iloc[:, 1:] >= 0).all(axis=1)]
    data_df = data_df.reset_index(drop=True)
    data_df_copy = data_df.copy()
    first_column = data_df.pop(data_df.columns[0])
    FEATURE_COLUMNS = data_df.columns
    data_df.insert(len(data_df.columns), first_column.name, first_column)
    TARGET_COLUMNS = "RiskPerformance"
    continuous_featues = ['MSinceOldestTradeOpen',
    'AverageMInFile',
    'NetFractionInstallBurden',
    'NetFractionRevolvingBurden',
    'MSinceMostRecentTradeOpen',
    'PercentInstallTrades',
    'PercentTradesWBalance',
    'NumTotalTrades',
    'MSinceMostRecentDelq',
    'NumSatisfactoryTrades',
    'PercentTradesNeverDelq',
    'ExternalRiskEstimate']
    data, numeric_columns, categorical_columns = preprocess_dataset(data_df, continuous_features=continuous_featues)
    data_df_copy = data.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)
    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, continuous_featues


def calculate_num_bins(num_unique_values, value_range):
    num_bins = min(6, int(np.log2(num_unique_values)) + 1)
    num_bins = min(num_bins, value_range)
    return num_bins

def preprocess_dataset(df, continuous_features=[], one_hot_encode=True, datasetName="Adult"):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()
    numeric_columns = []
    categorical_columns = []

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category' and col not in continuous_features:
            categorical_columns.append(col)
            if len(df[col].unique()) == 2:
                df[col] = label_encoder.fit_transform(df[col])
            elif one_hot_encode or (one_hot_encode and datasetName == "Adult" and col == 'race'):
                encoded_values = onehot_encoder.fit_transform(df[[col]])
                new_cols = [col + '_' + str(i) for i in range(encoded_values.shape[1])]
                encoded_df = pd.DataFrame(encoded_values.toarray(), columns=new_cols)
                df = pd.concat([df, encoded_df], axis=1)
                df.drop(col, axis=1, inplace=True)
        elif df[col].dtype == 'object' or df[col].dtype == 'category' and df[col].str.isnumeric().all() and col not in continuous_features:
            df[col] = df[col].astype(int) 
            categorical_columns.append(col)
        elif col in continuous_features:
            numeric_columns.append(col)
            num_unique_values = len(df[col].unique())
            value_range = df[col].max() - df[col].min()
            num_bins = calculate_num_bins(num_unique_values, value_range)
            bin_discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform', subsample=None)
            bins = bin_discretizer.fit_transform(df[[col]])
            df[col] = bins.astype(int)
        else:
            if len(df[col].unique()) > 2:
                numeric_columns.append(col)
    return df, numeric_columns, categorical_columns
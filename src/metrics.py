from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import main_cost_constrained_GCFEs, main_coverage_constrained_GCFEs_MIP
from test_utils import nice_numbers, dataset_feature_descriptions
from main import *

def find_saturation_point_and_max_d(total_costs, k_values):
    max_d = max(total_costs)
    saturation_point = k_values[total_costs.index(max_d)]
    return saturation_point, max_d

def filter_subdict(experiment, allowed_subkeys):
    result = {}
    for key, value in experiment.items():
        if isinstance(value, dict):
            filtered_subdict = {}
            for subkey, subvalue in value.items():
                if subkey in allowed_subkeys:
                    filtered_subdict[subkey] = subvalue
            result[key] = filtered_subdict
        else:
            result[key] = value
    return result

allowed_subkeys = ["Coverage", "Avg. distance", "Avg. path cost"]

def max_possible_distance_in_dataset(datasetName):
    _, FEATURE_COLUMNS, _, _, _, _,\
          _, _, one_hot_encode_features = load_dataset(datasetName=datasetName)
    feature_set = set()
    for feature in FEATURE_COLUMNS:
        if feature == 'target':
            continue
        if feature in one_hot_encode_features:
            # Extract the prefix to count only once
            prefix = feature.split('_')[0]
            feature_set.add(prefix)
        else:
            feature_set.add(feature)
    
    return np.sqrt(len(feature_set))

def kAUC(datasetName="Student", epsilon=0.5, group_identifier='sex', group_identifier_value=None, classifier='xgb',\
          bandwith_approch="mean_scotts_rule", upper_limit_for_k=10, lower_limit_range_for_d=None, upper_limit_range_for_d=None, steps=10, skip_distance_calculation=True,\
                     skip_fgce_calculation=True, skip_model_training=True, skip_bandwith_calculation=True, skip_graph_creation=True, representation=64, verbose=False):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}

    fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
			  FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon,\
                datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch, verbose=verbose,\
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
                skip_graph_creation=skip_graph_creation, representation=representation, skip_bandwith_calculation=skip_bandwith_calculation)
    fgce_init_dict = {"fgce": fgce, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
         "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
         "positive_points": positive_points, "FN": FN, "FN_negatives_by_group": FN_negatives_by_group, "node_connectivity": node_connectivity,\
              "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}

    if lower_limit_range_for_d is None:
        lower_limit_range_for_d = 0.1
    if upper_limit_range_for_d is None:
        upper_limit_range_for_d = np.max(distances) 
    elif upper_limit_range_for_d == "max_distance_dataset":
        upper_limit_range_for_d = max_possible_distance_in_dataset(datasetName)

    k_values = nice_numbers(1, upper_limit_for_k, steps, score='k')
    for cfes in k_values:
        auc_matrix[cfes] = {}
        results = {}
        
        d_values = nice_numbers(lower_limit_range_for_d, upper_limit_range_for_d, steps, score='d')
        for max_d in d_values:
            r = filter_subdict(main_cost_constrained_GCFEs(epsilon=epsilon, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                skip_model_training=skip_model_training, skip_fgce_calculation=skip_fgce_calculation, skip_graph_creation=skip_graph_creation,
                                max_d = max_d, cost_function = "max_vector_distance", k=cfes, k_selection_method="accross_all_ccs", fgce_init_dict=fgce_init_dict)[0], allowed_subkeys)
            r.pop("Node Connectivity")
            r.pop("Edge Connectivity")
            r.pop("Total coverage")
            r.pop("Graph Stats")
            r.pop("Time")
            results[max_d] = (r)

        group_keys = list(r.keys())
        saturation_points_g = {key: 0 for key in group_keys}
        coverage_till_now_g = {key: 0 for key in group_keys}
        
        group_coverages = {key: [] for key in group_keys}
        
        for d in results:
            for key in group_keys:
                cov = results[d][key]['Coverage']
                if cov > coverage_till_now_g[key]:
                    coverage_till_now_g[key] = cov
                    saturation_points_g[key] = d
                group_coverages[key].append(cov)
        
        saturation_points[cfes] = saturation_points_g
        cov_for_saturation_points[cfes] = coverage_till_now_g

        max_auc = auc(d_values, [100]*len(d_values))

        auc_matrix[cfes] = {}
        
        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(d_values, group_coverages_array) / max_auc, 2)
            auc_matrix[cfes][key] = normalized_auc
    
    return saturation_points, cov_for_saturation_points, auc_matrix 


def dAUC(datasetName="Student", epsilon=0.7, group_identifier='sex', group_identifier_value='None', classifier='xgb',\
        upper_limit_for_k=10, lower_limit_range_for_d=None, upper_limit_range_for_d=None, steps=10, skip_fgce_calculation=True, skip_model_training=True,\
        skip_distance_calculation=True, skip_bandwith_calculation=True, skip_graph_creation=True, representation=64, bandwith_approch='mean_scotts_rule', verbose=False):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}
    cov_for_saturation_points = {}

    fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
			  FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon,\
                datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch, verbose=verbose,\
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
                skip_graph_creation=skip_graph_creation, representation=representation, skip_bandwith_calculation=skip_bandwith_calculation)
    fgce_init_dict = {"fgce": fgce, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
         "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
         "positive_points": positive_points, "FN": FN, "FN_negatives_by_group": FN_negatives_by_group, "node_connectivity": node_connectivity,\
              "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}
    
    if lower_limit_range_for_d is None:
        lower_limit_range_for_d = 0.1
    if upper_limit_range_for_d is None:
        upper_limit_range_for_d = np.max(distances) 
    elif upper_limit_range_for_d == "max_distance_dataset":
        upper_limit_range_for_d = max_possible_distance_in_dataset(datasetName)
    d_values = nice_numbers(lower_limit_range_for_d, upper_limit_range_for_d, steps, score='d')

    k_values =nice_numbers(1, upper_limit_for_k, steps, score='k')
    for d in d_values: 
        d = np.round(d, 2)
        auc_matrix[d] = {}
        results = {}
        for cfes in k_values:
            r = (filter_subdict(main_cost_constrained_GCFEs(epsilon=epsilon, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                    skip_model_training=skip_model_training, skip_fgce_calculation=skip_fgce_calculation, skip_graph_creation=skip_graph_creation,
                                    max_d = d, cost_function = "max_vector_distance", k=cfes, k_selection_method="accross_all_ccs", fgce_init_dict=fgce_init_dict)[0], allowed_subkeys))
            
            r.pop("Node Connectivity")
            r.pop("Edge Connectivity")
            r.pop("Total coverage")
            r.pop("Graph Stats")
            r.pop("Time")
            results[cfes] = (r)
        
        group_keys = list(r.keys())
        saturation_points_g = {key: 0 for key in group_keys}
        coverage_till_now_g = {key: 0 for key in group_keys}

        group_coverages = {key: [] for key in group_keys}

        for cfes in results:
            for key in group_keys:
                cov = results[cfes][key]['Coverage']
                if cov > coverage_till_now_g[key]:
                    coverage_till_now_g[key] = cov
                    saturation_points_g[key] = cfes
                group_coverages[key].append(cov)
            
        saturation_points[d] = saturation_points_g
        cov_for_saturation_points[d] = coverage_till_now_g

        max_auc = auc(k_values, [100]*len(k_values))

        auc_matrix[d] = {}
        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(k_values, group_coverages_array) / max_auc, 2)
            auc_matrix[d][key] = normalized_auc
        
    return saturation_points, cov_for_saturation_points, auc_matrix

def cAUC(datasetName="Student", group_identifier="sex", group_identifier_value=None, epsilon=0.5, k_values=None, coverages=None,\
         classifier='xgb', bandwith_approch='mean_scotts_rule', skip_model_training=True, skip_distance_calculation=True,\
            skip_graph_creation=True, skip_bandwith_calculation=True, representation=64, verbose=False):
    results = {coverage: {k: None for k in k_values} for coverage in coverages}

    fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
			  FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon,\
                datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch, verbose=verbose,\
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
                skip_graph_creation=skip_graph_creation, representation=representation, skip_bandwith_calculation=skip_bandwith_calculation)
    fgce_init_dict = {"fgce": fgce, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
         "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
         "positive_points": positive_points, "FN": FN, "FN_negatives_by_group": FN_negatives_by_group, "node_connectivity": node_connectivity,\
              "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}

    for coverage in coverages:
        for k in k_values:
            results[coverage][k] = main_coverage_constrained_GCFEs_MIP(epsilon=epsilon, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                skip_model_training=True, skip_graph_creation=True, skip_fgce_calculation=False, skip_distance_calculation=True,
                                cost_function = "max_vector_distance", k=k, cov=coverage, fgce_init_dict=fgce_init_dict, verbose=verbose)
    
    saturation_points_cov, y_values_cov, aucs_cov = {}, {}, {}
    for cov in results:
        total_costs_group = {}
        saturation_points, y_values, aucs = {}, {}, {}

        for k in results[cov]:
            for group in results[cov][k]:
                if group not in total_costs_group:
                    total_costs_group[group] = [results[cov][k][group]["Total Cost"]]
                else:
                    total_costs_group[group].append(results[cov][k][group]["Total Cost"])

        for group in total_costs_group:
            saturation_points[group], y_values[group] = find_saturation_point_and_max_d(total_costs_group[group], k_values)

        min_overal_cost = min([min(total_costs_group[group]) for group in total_costs_group])
        optimal_auc = auc(k_values, [min_overal_cost] * len(k_values))

        for group in total_costs_group:
            aucs[group] = auc(k_values, total_costs_group[group]) / optimal_auc

        saturation_points_cov[cov] = saturation_points
        y_values_cov[cov] = y_values
        aucs_cov[cov] = aucs

    return saturation_points_cov, y_values_cov, aucs_cov

def plot_k_or_dAUC(datasetName, saturation_points, cov_for_saturation_points, auc_matrix, score='k',\
        expand_left_x_axis=0, expand_right_x_axis=1, expand_bottom_y_axis=0.5, expand_top_y_axis=0.5):
    x_values = list(auc_matrix.keys())
    group_keys = list(auc_matrix[x_values[0]].keys())
    x_values_g0 = [auc_matrix[x][group_keys[0]] for x in x_values]
    x_values_g1 = [auc_matrix[x][group_keys[1]] for x in x_values]

    sp_G0 = [saturation_points[x][group_keys[0]] for x in x_values]
    sp_G1 = [saturation_points[x][group_keys[1]] for x in x_values]

    max_cov_G0 = [np.round(cov_for_saturation_points[x][group_keys[0]],2) for x in x_values]
    max_cov_G1 = [np.round(cov_for_saturation_points[x][group_keys[1]],2) for x in x_values]

    sns.set(style="white")
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=x_values, y=x_values_g0, marker='o', color='mediumseagreen', label='Group 0', linewidth=2)
    sns.lineplot(x=x_values, y=x_values_g1, marker='s', color='coral', label='Group 1', linewidth=2)
    plt.xlabel(score, fontsize=20)
    plt.ylabel(f'{score.upper()}AUC Score', fontsize=20)
    plt.xticks(x_values)
    plt.legend(fontsize=16, framealpha=0.2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if score == 'k':
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}{datasetName}_kAUC_scores.pdf")
    else:
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}{datasetName}_dAUC_scores.pdf")
    plt.show()

    plt.figure(figsize=(9, 6))
    sns.lineplot(x=x_values, y=sp_G0, color='mediumseagreen', label='Group 0', marker='o', linewidth=2, alpha=0.7)
    sns.lineplot(x=x_values, y=sp_G1, color='coral', label='Group 1', marker='s', linewidth=2, alpha=0.7)
    
    ## get the axis limit from plt
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(x_min-expand_left_x_axis, max(x_values) +expand_right_x_axis)
    plt.ylim(y_min-expand_bottom_y_axis, y_max+expand_top_y_axis)

    x_min, x_max = 0, 100  
    extend_x = 0 
    extend_y = 0  

    for i, (x, sp0, max_cov0, sp1, max_cov1) in enumerate(zip(x_values, sp_G0, max_cov_G0, sp_G1, max_cov_G1)):
        if sp0 > sp1:
            offset0 = (10, 10)
            offset1 = (10, -20)
        elif sp0 < sp1:
            offset0 = (10, -20)
            offset1 = (10, 10)
        else:
            if max_cov0 > max_cov1:
                offset0 = (10, 10)
                offset1 = (10, -20)
            else:
                offset0 = (10, -20)
                offset1 = (10, 10)

        annotation_x_max_0 = i* offset0[0]   
        annotation_y_max_0 = i* offset0[1] 
        if score == 'k':
            if annotation_x_max_0 > x_max:
                extend_x = extend_x+1 
                extend_x = extend_x+1 
        elif score == 'd':
            if annotation_x_max_0 > x_max:
                extend_x = extend_x+1 
                extend_x = extend_x+1 
        if annotation_y_max_0 > max(sp_G0):
            extend_y = extend_y+1 

        plt.annotate(f'{max_cov0}', (x, sp0), textcoords="offset points", xytext=offset0, \
                     ha='center', color='mediumseagreen', weight='bold')
        plt.annotate(f'{max_cov1}', (x, sp1), textcoords="offset points", xytext=offset1, \
                     ha='center', color='coral', weight='bold')

    plt.legend(fontsize=16, framealpha=0.2)    
    plt.xticks(x_values, fontsize=20)
    plt.yticks(fontsize=20)

    if score == 'k':
        plt.ylabel('Saturation Point: sp(k)', fontsize=20)
        plt.xlabel('k', fontsize=20)
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}{datasetName}_kAUC_sp_cov.pdf", dpi=300, bbox_inches=None)
    else:
        plt.ylabel('Saturation Point: sp(d)', fontsize=20)
        plt.xlabel('d', fontsize=20)
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}{datasetName}_dAUC_sp_cov.pdf", pad_inches=0.1)

    plt.show()


# =====================================================================================================================
# =====================================================================================================================
#                 		 					ATTRIBUTION
# =====================================================================================================================
# =====================================================================================================================
def generate_recourse_rules_per_wcc(dataframe, results, FEATURE_COLUMNS, datasetName):
    processed_features = set()
    group_actions = {}

    # Get the aWCCs for each group
    aWCCs_per_group = {}
    for group_id, group_stats in results['Graph Stats'].items():
        aWCCs = set()
        r0 = results[group_id]
        for fn in r0:
            if fn in ['Coverage', 'Avg. distance', 'Avg. path cost', 'Median distance', 'Median path cost']:
                continue
            aWCCs.add(r0[fn]['cfe_cc'])
        aWCCs_per_group[group_id] = aWCCs
    
    for group_id, group_stats in results.items():
        if group_id in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats", "Time"]:
            continue
        total_action_for_group = {}
        for fn_id, cfe_details in group_stats.items():
            try:
                fn_id = int(fn_id)
            except ValueError:
                continue  
            wcc = cfe_details['cfe_cc']
            if wcc not in aWCCs_per_group[group_id]:  # Skip non-aWCCs
                continue

            fn_vector = dataframe.loc[fn_id, FEATURE_COLUMNS]
            cfe_vector = dataframe.loc[cfe_details['CFE_name'], FEATURE_COLUMNS]

            actions = []
            for col in FEATURE_COLUMNS:
                if fn_vector[col] != cfe_vector[col]:
                    # handle one-hot encoded features
                    if datasetName != "Heloc" and any(col.startswith(key) for key in dataset_one_hot_mapping[datasetName].keys()):
                        main_feature, value = col.rsplit('_', 1)
                        if main_feature in processed_features:
                            continue
                        else:
                            processed_features.add(main_feature)
                            actions.append((main_feature, fn_vector[col], cfe_vector[col]))
                    else:
                        if col in dataset_feature_descriptions[datasetName]:
                            actions.append((dataset_feature_descriptions[datasetName][col], fn_vector[col], cfe_vector[col]))
                        else:
                            actions.append((col, fn_vector[col], cfe_vector[col]))
            if wcc not in total_action_for_group:
                total_action_for_group[wcc] = [actions]
            else:
                total_action_for_group[wcc].append(actions)

        group_actions[group_id] = total_action_for_group
    return group_actions        
    
def generate_recourse_rules(dataframe, results, FEATURE_COLUMNS, datasetName):
    processed_features = set()

    group_actions = {}
    
    for group_id, group_stats in results.items():
        if group_id in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats", "Time"]: 
            continue

        total_action_for_group = []
        
        for fn_id, cfe_details in group_stats.items():
            try:
                fn_id = int(fn_id)
            except ValueError:
                continue  
            
            fn_vector = dataframe.loc[fn_id, FEATURE_COLUMNS]
            cfe_vector = dataframe.loc[cfe_details['CFE_name'], FEATURE_COLUMNS]
            actions = []
            for col in FEATURE_COLUMNS:
                if fn_vector[col] != cfe_vector[col]:
                    # handle one-hot encoded features
                    if datasetName != "Heloc" and any(col.startswith(key) for key in dataset_one_hot_mapping[datasetName].keys()):
                        main_feature, value = col.rsplit('_', 1)
                        if main_feature in processed_features:
                            continue
                        else:
                            processed_features.add(main_feature)
                            actions.append((main_feature, fn_vector[col], cfe_vector[col]))
                    else:
                        if col in dataset_feature_descriptions[datasetName]:
                            actions.append((dataset_feature_descriptions[datasetName][col], fn_vector[col], cfe_vector[col]))
                        else:
                            actions.append((col, fn_vector[col], cfe_vector[col]))                        
            
            total_action_for_group.append(actions)

        group_actions[group_id] = total_action_for_group
    return group_actions

def sort_actions_by_frequency_per_wcc(actions_for_group):
    action_frequency = {}
    action_frequency_increment = {}
    subgroup_actions = {}
    for wcc, groupofactions in actions_for_group.items():
        subgroup_actions[wcc] = len(groupofactions)
        action_frequency[wcc] = {}
        action_frequency_increment[wcc] = {}
        for action in groupofactions:
            for a in action:
                if a[0] in action_frequency[wcc]:
                    action_frequency[wcc][a[0]] += 1
                else:
                    action_frequency[wcc][a[0]] = 1

                if a[0] in action_frequency_increment[wcc]:
                    action_frequency_increment[wcc][a[0]] += abs(a[1] - a[2])
                else:
                    action_frequency_increment[wcc][a[0]] = abs(a[1] - a[2])

        action_frequency[wcc] = {k: (v / subgroup_actions[wcc]) *100 for k, v in action_frequency[wcc].items()}
        action_frequency_increment[wcc] = {k: (v / subgroup_actions[wcc]) *100 for k, v in action_frequency_increment[wcc].items()}

        action_frequency[wcc] = dict(sorted(action_frequency[wcc].items(), key=lambda item: item[1], reverse=True))
        action_frequency_increment[wcc] = dict(sorted(action_frequency_increment[wcc].items(), key=lambda item: item[1], reverse=True))
    return action_frequency, action_frequency_increment

def sort_actions_by_frequency(actions_for_group):
    action_frequency = {}
    action_frequency_increment = {}
    for groupofactions in actions_for_group:
        for action in groupofactions:
            if action[0] in action_frequency:
                action_frequency[action[0]] += 1
            else:
                action_frequency[action[0]] = 1

            if action[0] in action_frequency_increment:
                action_frequency_increment[action[0]] += abs(action[1] - action[2])
            else:
                action_frequency_increment[action[0]] = abs(action[1] - action[2])

    total_actions = len(actions_for_group)
    action_frequency = {k: (v / total_actions) *100 for k, v in action_frequency.items()}
    action_frequency_increment = {k: (v / total_actions) *100 for k, v in action_frequency_increment.items()}


    action_frequency = dict(sorted(action_frequency.items(), key=lambda item: item[1], reverse=True))
    action_frequency_increment = dict(sorted(action_frequency_increment.items(), key=lambda item: item[1], reverse=True))
    return action_frequency, action_frequency_increment

def plot_feature_frequency(dataset_name, action_frequency_g0, action_frequency_g1, sx, sy, freq_threshold=None):
    filtered_keys_g0 = set(key for key, value in action_frequency_g0.items() if freq_threshold is None or value > freq_threshold)
    filtered_keys_g1 = set(key for key, value in action_frequency_g1.items() if freq_threshold is None or value > freq_threshold)
    filtered_keys = filtered_keys_g0.union(filtered_keys_g1)
    
    # Sort keys based on the maximum value for both groups
    sorted_keys = sorted(filtered_keys, key=lambda x: max(action_frequency_g0.get(x, 0), action_frequency_g1.get(x, 0)), reverse=True)
    
    y_combined = range(len(sorted_keys))                 
    bar_width = 0.3

    plt.barh(y_combined, [action_frequency_g0.get(key, 0) for key in sorted_keys], height=bar_width, color='mediumseagreen', align='center')
    plt.barh([y + bar_width for y in y_combined], [action_frequency_g1.get(key, 0) for key in sorted_keys], height=bar_width, color='coral', align='center')
    plt.ylabel('Attribute Description', fontsize=12)
    plt.xlabel('Frequency (%)', fontsize=12)
    plt.legend(['Group 0', 'Group 1'])
    plt.yticks([y + bar_width / 2 for y in y_combined], [dataset_feature_descriptions[dataset_name].get(key, key) for key in sorted_keys], fontsize=12)
    plt.xticks(fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    fig_size = (sx, sy) 
    plt.gcf().set_size_inches(fig_size)
    plt.tight_layout()
    plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{dataset_name}{sep}figs{sep}attribution.pdf")
    plt.show()

def plot_feature_frequency_per_wcc(datasetName, action_frequency_g0, action_frequency_g1, sx, sy, params_size, freq_threshold=None):
    plt.style.use('seaborn-muted')
    plt.rcParams.update({"font.family": "serif"})

    sorted_action_frequency_g0 = dict(sorted(action_frequency_g0.items()))
    
    for wcc, action_frequency_g0_wcc in sorted_action_frequency_g0.items():
        print(f"WCC: {wcc}")
        filtered_keys_g0 = set(key for key, value in action_frequency_g0_wcc.items() if freq_threshold is None or value > freq_threshold)
        sorted_keys_g0 = sorted(filtered_keys_g0, key=lambda x: action_frequency_g0_wcc.get(x, 0), reverse=True)
        fig, ax = plt.subplots(figsize=(sx, sy))
        color_g0 = '#CC6677'  
        ax.barh(sorted_keys_g0, [action_frequency_g0_wcc.get(key, 0) for key in sorted_keys_g0], 
                height=0.3, color=color_g0, align='center')

        ax.set_yticks(range(len(sorted_keys_g0)))
        ax.set_yticklabels([dataset_feature_descriptions[datasetName].get(key, key) for key in sorted_keys_g0], fontsize=params_size)
        ax.set_ylabel('Attribute', fontsize=18)
        ax.set_xlabel('ACF', fontsize=18)
        ax.tick_params(axis='x', labelsize=params_size)
        ax.tick_params(axis='y', labelsize=params_size)
        ax.invert_yaxis()
        # ax.grid(axis='x', linestyle='--', alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}attribution_{wcc}.pdf", dpi=300)
        plt.show()

    sorted_action_frequency_g1 = dict(sorted(action_frequency_g1.items()))

    for wcc, action_frequency_g1_wcc in sorted_action_frequency_g1.items():
        print(f"WCC: {wcc}")
        filtered_keys_g1 = set(key for key, value in action_frequency_g1_wcc.items() if freq_threshold is None or value > freq_threshold)
        sorted_keys_g1 = sorted(filtered_keys_g1, key=lambda x: action_frequency_g1_wcc.get(x, 0), reverse=True)
        fig, ax = plt.subplots(figsize=(sx, sy))
        color_g1 = '#882255'  
        ax.barh(sorted_keys_g1, [action_frequency_g1_wcc.get(key, 0) for key in sorted_keys_g1], 
                height=0.3, color=color_g1, align='center')
        ax.set_yticks(range(len(sorted_keys_g1)))
        ax.set_yticklabels([dataset_feature_descriptions[datasetName].get(key, key) for key in sorted_keys_g1], fontsize=params_size)
        ax.set_ylabel('Attribute', fontsize=18)
        ax.set_xlabel('ACF', fontsize=18)
        ax.tick_params(axis='x', labelsize=params_size)
        ax.tick_params(axis='y', labelsize=params_size)
        ax.invert_yaxis()
        # ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}attribution_{wcc}.pdf", dpi=300)
        plt.show()

def attribution_analysis(datasetName='Adult', epsilon=0.4, group_identifier='sex', group_identifier_value=None,\
        classifier="xgb", skip_model_training=True, bandwith_approch="mean_scotts_rule", skip_bandwith_calculation=True,\
        max_d=1.05, cost_function="max_vector_distance", skip_distance_calculation=True, skip_graph_creation=True,\
        k=12, k_selection_method="accross_all_ccs", skip_fgce_calculation=True, verbose=False,\
        per_group_per_subgroup="per_group", freq_threshold=50, x_axis_size=8, y_axis_size=6):

    results, data, _, _, _, _, _, _,_ = \
            main_cost_constrained_GCFEs(epsilon=epsilon, datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, group_identifier_value=group_identifier_value,\
            skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation, skip_bandwith_calculation=skip_bandwith_calculation,\
              skip_fgce_calculation=skip_fgce_calculation, skip_graph_creation=skip_graph_creation, bandwith_approch=bandwith_approch,\
            max_d = max_d, cost_function = cost_function, k=k, k_selection_method=k_selection_method, verbose=verbose)
    data = data.iloc[:, :-1]
    group_ids = []
    for group_id, _ in results.items():
        if group_id in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats"]:
            continue
        group_ids.append(group_id)
    if per_group_per_subgroup == "per_group":
        actions = generate_recourse_rules(data, results, data.columns, datasetName)
        action_frequency_g0, _ = sort_actions_by_frequency(actions[group_ids[0]])
        action_frequency_g1, _ = sort_actions_by_frequency(actions[group_ids[1]])
    elif per_group_per_subgroup == "per_subgroup":
        actions = generate_recourse_rules_per_wcc(data, results, data.columns, datasetName)
        action_frequency_g0, _ = sort_actions_by_frequency_per_wcc(actions[group_ids[0]])
        action_frequency_g1, _ = sort_actions_by_frequency_per_wcc(actions[group_ids[1]])
    return action_frequency_g0, action_frequency_g1
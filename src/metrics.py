from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import main_cost_constrained_GCFEs, main_coverage_constrained_GCFEs_MIP
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

def kAUC(datasetName="Student", epsilon=0.5, tp=0.5, td=0.001, group_identifier='sex', group_identifier_value=None, classifier='xgb',\
          bandwith_approch="mean_scotts_rule", upper_limit_for_k=10, lower_limit_range_for_d=0.1, steps=10, skip_distance_calculation=True,\
                     skip_fgce_calculation=True, skip_model_training=True, skip_bandwith_calculation=True, skip_graph_creation=True, representation=64):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}

    max_possible_distance_for_these_features = max_possible_distance_in_dataset(datasetName)
    d_step = np.round(((max_possible_distance_for_these_features - 0.1) /steps), 1)

    fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
			  FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon, tp=tp, td=td,\
                datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch,\
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_bandwith_calculation,\
                skip_graph_creation=skip_graph_creation, representation=representation)
    fgce_init_dict = {"fgce": fgce, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
         "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
         "positive_points": positive_points, "FN": FN, "FN_negatives_by_group": FN_negatives_by_group, "node_connectivity": node_connectivity,\
              "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}

    k_values = np.linspace(1, upper_limit_for_k, steps)
    k_values_int = np.round(k_values).astype(int)
    for cfes in k_values_int:
        auc_matrix[cfes] = {}
        results = {}
        
        for max_d in np.arange(lower_limit_range_for_d, max_possible_distance_for_these_features, d_step):
            r = filter_subdict(main_cost_constrained_GCFEs(epsilon=epsilon, tp=tp, td=td, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                skip_model_training=skip_model_training, skip_fgce_calculation=skip_fgce_calculation, skip_graph_creation=skip_graph_creation,
                                max_d = max_d, cost_function = "max_vector_distance", k=cfes, k_selection_method="greedy_accross_all_ccs", fgce_init_dict=fgce_init_dict)[0], allowed_subkeys)
            r.pop("Node Connectivity")
            r.pop("Edge Connectivity")
            r.pop("Total coverage")
            r.pop("Graph Stats")
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

        x = list(np.arange(lower_limit_range_for_d, max_possible_distance_for_these_features, d_step))
        max_auc = auc(x, [100]*len(x))

        auc_matrix[cfes] = {}
        
        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(x, group_coverages_array) / max_auc, 2)
            auc_matrix[cfes][key] = normalized_auc
    
    return saturation_points, cov_for_saturation_points, auc_matrix 


def dAUC(datasetName="Student", epsilon=0.7, tp=0.5, td=0.001, group_identifier='sex', group_identifier_value='None', classifier='xgb', upper_limit_for_k=10,\
          steps=10, skip_fgce_calculation=True, skip_model_training=True, skip_distance_calculation=True, skip_bandwith_calculation=True, skip_graph_creation=True, representation=64,
          bandwith_approch='mean_scotts_rule'):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}
    cov_for_saturation_points = {}

    max_possible_distance_for_these_features = max_possible_distance_in_dataset(datasetName)
    d_steps = np.round(np.linspace(0.1, max_possible_distance_for_these_features, num=steps), 1)

    fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
			  FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon=epsilon, tp=tp, td=td,\
                datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch,\
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
                skip_graph_creation=skip_graph_creation, skip_bandwith_calculation=skip_bandwith_calculation, representation=representation)
    fgce_init_dict = {"fgce": fgce, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
         "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
         "positive_points": positive_points, "FN": FN, "FN_negatives_by_group": FN_negatives_by_group, "node_connectivity": node_connectivity,\
              "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}
    
    k_values = np.linspace(1, upper_limit_for_k, steps)
    k_values_int = np.round(k_values).astype(int)
    for d in d_steps: 
        d = np.round(d, 2)
        auc_matrix[d] = {}
        results = {}
        for cfes in k_values_int:
            r = (filter_subdict(main_cost_constrained_GCFEs(epsilon=epsilon, tp=tp, td=td, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                    skip_model_training=skip_model_training, skip_fgce_calculation=skip_fgce_calculation, skip_graph_creation=skip_graph_creation,
                                    max_d = d, cost_function = "max_vector_distance", k=cfes, k_selection_method="greedy_accross_all_ccs", fgce_init_dict=fgce_init_dict)[0], allowed_subkeys))
            
            r.pop("Node Connectivity")
            r.pop("Edge Connectivity")
            r.pop("Total coverage")
            r.pop("Graph Stats")
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

        max_auc = auc(k_values_int, [100]*len(k_values_int))

        auc_matrix[d] = {}
        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(k_values_int, group_coverages_array) / max_auc, 2)
            auc_matrix[d][key] = normalized_auc
        
    return saturation_points, cov_for_saturation_points, auc_matrix

def cAUC(datasetName="Student", group_identifier="sex", group_identifier_value=None, epsilon=0.5, k_values=None, coverages=None,\
         tp=0.5, td=0.001, classifier='xgb', bandwith_approch='mean_scotts_rule', skip_model_training=True, skip_distance_calculation=True,\
            skip_graph_creation=True, skip_bandwith_calculation=True, representation=64):
    results = {coverage: {k: None for k in k_values} for coverage in coverages}

    fgce, graph, distances, data, data_np, data_df_copy, attr_col_mapping, normalized_group_identifer_value, numeric_columns, positive_points,\
			  FN, FN_negatives_by_group, node_connectivity, edge_connectivity, feasibility_constraints  = initialize_FGCE(epsilon=epsilon, tp=tp, td=td,\
                datasetName=datasetName, group_identifier=group_identifier, classifier=classifier, bandwith_approch=bandwith_approch,\
                group_identifier_value=group_identifier_value, skip_model_training=skip_model_training, skip_distance_calculation=skip_distance_calculation,\
                skip_graph_creation=skip_graph_creation, representation=representation)
    fgce_init_dict = {"fgce": fgce, "graph": graph, "distances": distances, "data": data, "data_np": data_np, "data_df_copy": data_df_copy,\
         "attr_col_mapping": attr_col_mapping, "normalized_group_identifer_value": normalized_group_identifer_value, "numeric_columns": numeric_columns,\
         "positive_points": positive_points, "FN": FN, "FN_negatives_by_group": FN_negatives_by_group, "node_connectivity": node_connectivity,\
              "edge_connectivity": edge_connectivity, "feasibility_constraints": feasibility_constraints}

    for coverage in coverages:
        for k in k_values:
            results[coverage][k] = main_coverage_constrained_GCFEs_MIP(epsilon=epsilon, tp=0.6, td=0.001, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                skip_model_training=True, skip_graph_creation=True, skip_fgce_calculation=False, skip_distance_calculation=True,
                                cost_function = "max_vector_distance", k=k, cov=coverage, fgce_init_dict=fgce_init_dict)
    
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

def plot_k_or_dAUC(datasetName, saturation_points, cov_for_saturation_points, auc_matrix, score='k'):
    x_values = list(auc_matrix.keys())
    group_keys = list(auc_matrix[x_values[0]].keys())
    x_values_g0 = [auc_matrix[x][group_keys[0]] for x in x_values]
    x_values_g1 = [auc_matrix[x][group_keys[1]] for x in x_values]

    min_y_value = min(min(x_values_g0), min(x_values_g1))
    max_y_value = max(max(x_values_g0), max(x_values_g1))

    sp_G0 = [saturation_points[x][group_keys[0]] for x in x_values]
    sp_G1 = [saturation_points[x][group_keys[1]] for x in x_values]

    max_cov_G0 = [np.round(cov_for_saturation_points[x][group_keys[0]],2) for x in x_values]
    max_cov_G1 = [np.round(cov_for_saturation_points[x][group_keys[1]],2) for x in x_values]

    sns.set(style="white")

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=x_values, y=x_values_g0, marker='o', color='mediumseagreen', label='Group 0')
    sns.lineplot(x=x_values, y=x_values_g1, marker='s', color='coral', label='Group 1')
    plt.xlabel(score, fontsize=20)
    plt.ylabel(f'{score.upper()}AUC Score', fontsize=20)
    plt.ylim(min_y_value - 0.2, max_y_value + 0.1)
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



    plt.figure(figsize=(8, 6))
    sns.lineplot(x=x_values, y=sp_G0, color='mediumseagreen', label='Group 0', marker='o')
    sns.lineplot(x=x_values, y=sp_G1, color='coral', label='Group 1', marker='s')

    # Set initial x-axis limit
    plt.xlim(min(x_values) - 0.1, max(x_values) + (1 if score == 'k' else 0.5))
    plt.ylim(min(min(sp_G0), min(sp_G1)) - (0.45 if score == 'k' else 2.2),
            max(max(sp_G0), max(sp_G1)) + (1 if score == 'k' else 3.2))

    x_min, x_max = 0, 100  
    extend_x = 0 
    extend_y = 0  

    for i, (x, sp0, max_cov0, sp1, max_cov1) in enumerate(zip(x_values, sp_G0, max_cov_G0, sp_G1, max_cov_G1)):
        if sp0 > sp1:
            offset0 = (15, 20)
            offset1 = (15, -20)
        elif sp0 < sp1:
            offset0 = (15, -20)
            offset1 = (15, 20)
        else:
            if max_cov0 > max_cov1:
                offset0 = (15, 20)
                offset1 = (15, -20)
            else:
                offset0 = (15, -20)
                offset1 = (15, 20)

        annotation_x_max_0 = i* offset0[0]   
        annotation_y_max_0 = i* offset0[1] 
        if x > 10 and score == 'k':
            if annotation_x_max_0 > x_max:
                extend_x = extend_x+1 
                extend_x = extend_x+1 
        elif score == 'd':
            if annotation_x_max_0 > x_max:
                extend_x = extend_x+1 
                extend_x = extend_x+1 
        if annotation_y_max_0 > max(sp_G0):
            extend_y = extend_y+1 

        plt.annotate(f'{max_cov0}', (x, sp0), textcoords="offset points", xytext=offset0, ha='center', color='mediumseagreen')
        plt.annotate(f'{max_cov1}', (x, sp1), textcoords="offset points", xytext=offset1, ha='center', color='coral')

    if score == 'k':
        if extend_x > 0:
            plt.xlim(x_min, max(x_values) + extend_x)
    if score == 'd':
        if extend_y > 0:
            plt.ylim(min(min(sp_G0), min(sp_G1)) - extend_y,
                max(max(sp_G0), max(sp_G1)) + extend_y) 

    plt.xticks(x_values)
    legend = plt.legend(title='Numbers indicate Maximum Coverage', loc='best', fontsize=16, framealpha=0.3)
    legend.get_title().set_fontsize(16) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    legend.get_title().set_ha('center')

    plt.ylabel('Saturation Point', fontsize=20)

    if score == 'k':
        plt.xlabel('k', fontsize=20)
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}{datasetName}_kAUC_sp_cov.pdf", bbox_inches='tight')
    else:
        plt.xlabel('d', fontsize=20)
        plt.savefig(f"{FGCE_DIR}{sep}tmp{sep}{datasetName}{sep}figs{sep}{datasetName}_dAUC_sp_cov.pdf", bbox_inches='tight')

    plt.show()
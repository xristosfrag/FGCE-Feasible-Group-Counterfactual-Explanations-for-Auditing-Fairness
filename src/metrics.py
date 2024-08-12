from sklearn.metrics import auc
import numpy as np
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

def kAUC(datasetName, epsilon, group_identifier, group_identifier_value, upper_limit_for_k, lower_limit_range_for_d, max_possible_distance_for_these_features, step):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}

    for cfes in range(1, upper_limit_for_k):
        auc_matrix[cfes] = {}
        results = {}
        
        for max_d in np.arange(lower_limit_range_for_d, max_possible_distance_for_these_features, step):
            r = filter_subdict(main_cost_constrained_GCFEs(epsilon=epsilon, tp=0.6, td=0.001, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                skip_model_training=True, skip_gcfe_calculation=False, skip_graph_creation=True,
                                max_d = max_d, cost_function = "max_vector_distance", k=cfes, k_selection_method="greedy_accross_all_ccs")[0], allowed_subkeys)
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

        x = list(np.arange(lower_limit_range_for_d, max_possible_distance_for_these_features, step))
        max_auc = auc(x, [100]*len(x))

        auc_matrix[cfes] = {}
        
        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(x, group_coverages_array) / max_auc, 2)
            auc_matrix[cfes][key] = normalized_auc
    
    return saturation_points, cov_for_saturation_points, auc_matrix    

def dAUC(datasetName, epsilon, group_identifier, group_identifier_value, upper_limit_for_k, max_possible_distance_for_these_features, step):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}
    cov_for_saturation_points = {}

    for d in list([np.arange(0.1, max_possible_distance_for_these_features, step)])[0]:
        d = np.round(d, 2)
        auc_matrix[d] = {}
        results = {}
        for cfes in np.arange(1, upper_limit_for_k, 1):
            r = (filter_subdict(main_cost_constrained_GCFEs(epsilon=epsilon, tp=0.6, td=0.001, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                    skip_model_training=True, skip_gcfe_calculation=False, skip_graph_creation=True,
                                    max_d = d, cost_function = "max_vector_distance", k=cfes, k_selection_method="greedy_accross_all_ccs")[0], allowed_subkeys))
            
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

        x = list(np.arange(1, upper_limit_for_k, 1))
        max_auc = auc(x, [100]*len(x))

        auc_matrix[d] = {}
        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(x, group_coverages_array) / max_auc, 2)
            auc_matrix[d][key] = normalized_auc
        
    return saturation_points, cov_for_saturation_points, auc_matrix

def cAUC(datasetName, group_identifier, group_identifier_value, epsilon, k_values, coverages):
    results = {coverage: {k: None for k in k_values} for coverage in coverages}

    for coverage in coverages:
        for k in k_values:
            results[coverage][k] = main_coverage_constrained_GCFEs_MIP(epsilon=epsilon, tp=0.6, td=0.001, datasetName=datasetName, group_identifier=group_identifier, group_identifier_value=group_identifier_value,
                                skip_model_training=True, skip_graph_creation=True, skip_gcfe_calculation=False, skip_distance_calculation=True,
                                cost_function = "max_vector_distance", k=k, cov=coverage)
    
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
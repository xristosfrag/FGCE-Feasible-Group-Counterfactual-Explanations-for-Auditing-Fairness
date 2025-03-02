from math import inf
import numpy as np
from tqdm import tqdm
import utils
import networkx as nx
from pulp import LpMaximize, LpMinimize, LpBinary, LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from collections import deque
import heapq
import time

random_seed = 482
np.random.seed(random_seed)

class FGCE:
	def __init__(self, data, kernel_obj, feature_columns, target_column, epsilon, clf=None):
		self._data = data
		self._epsilon = epsilon
		self._clf = clf
		self._kernel_obj = kernel_obj
		self.set_feature_names(feature_columns, target_column)
		self._Graph = None
		self.I = None
	
	def set_candidates(self, I):
		self.I = I

	def set_feature_names(self, features, target):
		self.FEATURE_COLUMNS = features
		self.TARGET_COLUMN = target

	def make_graph(self, distances, kernel, feasibilitySet, epsilon, datasetName=""):
		"""
		Make the graph using given data points

		# Arguments:
		----------
		- distances: (np.array)
			distances between data points
		- kernel: (Kernel_obj)
			kernel object
		- feasibilitySet: (FeasibilitySet object)
			feasibility set constraints
		- epsilon: (float)
			epsilon value
		- datasetName: (str)
			name of the dataset

		# Returns:
		----------
		- Graph: (networkx.Graph)
			constructed graph
		- distances: (np.array)
			distances between data points
		- densities: (np.array)
			densities of data points
		"""
		print("Constructing graph...")
		self._Graph, distances, densities = utils.make_graph(self.X, distances, kernel, feasibilitySet, epsilon, datasetName)
		return self._Graph, distances, densities
	
	def set_graph(self, graph):
		"""
		Sets the graph

		# Arguments:
		----------
		- graph: (networkx.Graph)
			graph
		"""
		self._Graph = graph

	def get_personalized_candidates(self, source_index, feasibility_constraints):
		"""
		Get personalized candidates for the source point
		
		# Arguments:
		----------
		- source_index: (int)
			index of the source point
		- feasibility_constraints: (FeasibilitySet object)
			feasibility set constraints
		
		# Returns:
		----------
		- candidates: (dict)
			personalized candidates
		"""
		candidates = {}
		source_point = self._data[source_index, :-1]

		for x_id, x in self.I.items():
			dest_point = x
			if feasibility_constraints.check_constraints(source_point, dest_point[:-1]):
				candidates[x_id] = x
		return candidates
	
	def compute_recourse(self, graph, source, feasibility_constraints, personalized=True):
		"""	
		Computes all shortest paths from source to all candidate points and know the minimum path cost to source point.

		# Arguments:
		- graph: (networkx.Graph)
			graph
		- source: (int)
			index of the source point
		- feasibility_constraints: (FeasibilitySet object)
			feasibility set constraints
		- personalized: (bool)
			whether to use personalized candidates or not
		
		# Returns:
			- shortest_paths: (dict)
				shortest paths from source to all candidate points
			- min_target: (int)
				index of the candidate point with minimum path cost to source point. CFE name
		"""
		assert (self._Graph is not None)
		
		if personalized:
			personalized_I = self.get_personalized_candidates(source, feasibility_constraints)
		else:
			personalized_I = self.I

		def find_shortest_paths_and_costs(graph, source, candidates, weight='wij'):
			"""
			Find shortest paths and costs from source to all candidate points.

			# Arguments:
			----------
			- graph: (networkx.Graph)
				graph
			- source: (int)
				source point
			- candidates: (dict)
				candidate points
			- weight: (str)
				weight to use for the shortest path computation
			
			# Returns:
			----------
			- shortest_paths_info: (dict)
				shortest paths from source to all candidate points
			- min_target_id: (int)
				index of the candidate point with minimum path cost to source point. CFE name
			"""
			shortest_paths_info = {}
			try:
				path_cost_wij, paths = nx.single_source_dijkstra(graph, source, weight=weight)
				candidates_list = list(set(candidates.keys()))

				min_target_id = 0
				min_wij = float('inf')
				
				for target_id, path_to_target in paths.items():
					if target_id in candidates_list:
						path_cost_dist = 0
						path_length = len(path_to_target)

						for i in range(path_length-1):
							xi = path_to_target[i]
							xj = path_to_target[i+1]
							path_cost_dist += graph[xi][xj]['distance']

						path_cost = path_cost_wij[target_id]
						if path_cost < min_wij:
							min_wij = path_cost
							min_target_id = target_id

						shortest_paths_info[target_id] = {
							'vector': candidates[target_id],
							'path': path_to_target,
							'path_cost': path_cost,
							'path_cost_dist': path_cost_dist
						}
			except nx.NetworkXNoPath:
				return None, None

			return shortest_paths_info, min_target_id


		return find_shortest_paths_and_costs(graph, source, personalized_I, weight='wij')

# =========================================================================================================
# ============================cost-constrained group counterfactuals=======================================
# =========================================================================================================
# For the greedy coverage problem, the best solution is to use a BFS in order to find the candidate CFEs before selecting.
# However, when using other cost functions like the path cost distance wij, the number of hops of this shortest path, or any 
# other cost function that is based on the paths, we implemented it using shortest paths from each factual to 
# each candidate counterfactual.

	def greedy_selection(self, cfes, k=5):
		"""
		Greedy algorithm to select the CFES that cover the most uncovered individuals.

		# Arguments:
		----------
		- cfes: (dict)
			shortest paths from source to all candidate points
		- k: (int)
			maximum number of cfes to be returned

		# Returns:
		----------
		- selected_cfes: (list)
			selected cfes
		"""
		excluded_cfes=set()

		uncovered_individuals = set(individual for cfe in cfes.values() for individual in cfe['Covered_recourse_points'])
		selected_cfes = {}

		while uncovered_individuals and len(selected_cfes) < k:
			# Select the CFE that covers the most uncovered individuals, with the lowest path_cost in case of a tie
			# and is not in the excluded list.
			best_cfe = max(
				(item for item in cfes.items() if item[0] not in excluded_cfes),
				key=lambda item: (
					len(set(item[1]['Covered_recourse_points']) & uncovered_individuals),
					-item[1]['Decision_based_on_this_cost']
				),
				default=None  # In case all CFES are excluded
			)

			if not best_cfe:
				break
			selected_cfes[best_cfe[0]] = best_cfe[1]
			uncovered_individuals -= set(best_cfe[1]['Covered_recourse_points'])
			excluded_cfes.add(best_cfe[0])

		return selected_cfes
		
	def mip_selection(self, cfes, k):
		candidate_keys = list(cfes.keys())
		num_counterfactuals = len(candidate_keys)
		index_to_key_cfes = {j: candidate_keys[j] for j in range(num_counterfactuals)}
		factuals = list({i for cfe in cfes.values() for i in cfe['Covered_recourse_points']})
		num_factuals = len(factuals)
		
		cost_matrix = np.array([
			[cfes[index_to_key_cfes[j]]['Distance_cost'][factual] 
			if factual in cfes[index_to_key_cfes[j]]['Distance_cost'] else 0 
			for j in range(num_counterfactuals)]
			for factual in factuals
		])
		coverage_matrix = np.where(cost_matrix > 0, 1, 0)

		model = LpProblem("Cost_Constrained_FGCE", LpMaximize)

		# Constraint (5): Binary constraint for each candidate and assignment
		u = {j: LpVariable(f"u_{j}", cat="Binary") for j in range(num_counterfactuals)}
		r = {(i, j): LpVariable(f"r_{i}_{j}", cat="Binary")
			for i in range(num_factuals) for j in range(num_counterfactuals)}

		# Objective: maximize total coverage (sum of r over all factuals and candidates with coverage)
		model += lpSum(r[i, j] for i in range(num_factuals) for j in range(num_counterfactuals) if coverage_matrix[i, j] > 0)


		# Constraint (2): Limit the number of selected candidate CFs to at most k
		model += lpSum(u[j] for j in range(num_counterfactuals)) <= k

		# Constraint (3): Each factual instance is assigned to at most one candidate CF.
		for i in range(num_factuals):
			model += lpSum(r[i, j] for j in range(num_counterfactuals)) <= 1

		# Constraint (4): A factual can only be covered by candidate j if candidate j is selected
		for i in range(num_factuals):
			for j in range(num_counterfactuals):
				if coverage_matrix[i, j] > 0:
					model += r[i, j] <= u[j]

		model.solve(PULP_CBC_CMD(msg=False))

		if LpStatus[model.status] == "Optimal":
			selected_indices = [j for j in range(num_counterfactuals) if u[j].varValue > 0.5]
			selected_cfs = [index_to_key_cfes[j] for j in selected_indices]			
			return {key: cfes[key] for key in selected_cfs}
		else:
			print("Solver did not find an optimal solution. Status:", LpStatus[model.status])
			selected_cfs = {}

	def dijkstra_like_paths_to_positive_points(self, graph, start_node, candidate_cfes):
		"""
		Computes the shortest paths from the start node to all positive points.

		# Arguments:
		----------
		- graph: (networkx.Graph)
			graph
		- start_node: (int)
			start node
		- candidate_cfes: (list)
			positive class instances
		
		# Returns:
		----------
		- paths: (dict)
			shortest paths from start node to all positive points
		"""
		paths = {}
		pq = [] 
		heapq.heappush(pq, (0, 0, start_node, [start_node]))  # Heap elements are tuples of (weight, hops, node, path)
		visited = set()

		while pq and len(paths) < len(candidate_cfes):
			weight, hops, current_node, path = heapq.heappop(pq)

			if current_node in visited:
				continue
			visited.add(current_node)

			if current_node in candidate_cfes:
				paths[current_node] = (path, weight)

			for neighbor in graph.neighbors(current_node):
				if neighbor not in visited:
					edge_weight = graph.edges[current_node, neighbor]['wij']
					total_weight = weight + edge_weight
					heapq.heappush(pq, (total_weight, hops + 1, neighbor, path + [neighbor]))

		return paths
	
	def bfs(self, graph, start_node):
		"""
		Breadth-first search algorithm to find
		all reachable nodes from the start node.
		
		# Arguments:
		----------
		- graph: (networkx.Graph)
			graph
		- start_node: (int)
			start node
		
		# Returns:
		----------
		- visited: (set)
			visited nodes
		"""
		visited = set()
		queue = deque([start_node])
		visited.add(start_node)

		while queue:
			node = queue.popleft()
			for neighbor in graph.neighbors(node):
				if neighbor not in visited:
					visited.add(neighbor)
					queue.append(neighbor)

		return visited

	def compute_gcfes(self, subgroups, candidate_cfes, factuals, max_d, cost_function, k, distances, k_selection_method="accross_all_ccs", verbose=False, cfe_selection_method="greedy"):
		"""
		Computes the group CFES for each subgroup.

		# Parameters:
		----------
		- subgroups: (dict)
			subgroups
		- candidate_cfes: (list)
			positive class instances
		- factuals: (list)
			factual instances
		- max_d: (float)
			maximum distance
		- cost_function: (str)
			the method to use to compute d
		- k: (int)
			maximum number of cfes to be returned
		- distances: (dict)
			distances between instances
		- k_selection_method: (str)
			the method to use to select the CFES
		
		# Returns:
		----------
		- gcfes: (dict)
			group CFES for each subgroup
		- not_possible_to_cover_factuals_group: (dict)
			factual instances that are not possible to cover
		- time_fgces: (float)
			time taken to compute the group CFES
		"""
		gcfes = {}
		gcfes['stats'] = {}
		not_possible_to_cover_factuals_group = {}
		total_ccs_not_applicable = 0
		total_ccs_applicable = 0
		ccs_cfes_index = 0
		time_fgces = 0

		for subgroup_index, subgroup in subgroups.items():
			connected_components = list(nx.weakly_connected_components(subgroup))
			gcfes['stats'][subgroup_index] = {}
			gcfes['stats'][subgroup_index]['nodes'] = len(subgroup)
			gcfes['stats'][subgroup_index]['connected_components'] = len(connected_components)
			if verbose:
				print(f"Subgroup: {subgroup_index} - Nodes: {len(subgroup)}\nConnected components: {len(connected_components)}")

			ccs_cfes = {}			
			ccs_not_applicable = 0
			not_possible_to_cover_factuals_group[subgroup_index] = {}

			for connected_component in connected_components:
				cfes = {}
				subgraph = subgroup.subgraph(connected_component)
				candidate_cfes_in_subgraph = set(connected_component) & set(candidate_cfes)
				factuals_in_subgraph = set(connected_component) & set(factuals)

				if not factuals_in_subgraph or not candidate_cfes_in_subgraph:
					ccs_not_applicable += 1
					total_ccs_not_applicable += 1
					ccs_cfes_index += 1
					continue
				total_ccs_applicable += 1
				gcfes['stats'][subgroup_index][ccs_cfes_index] = {}
				gcfes['stats'][subgroup_index][ccs_cfes_index]['nodes'] = len(connected_component)
				gcfes['stats'][subgroup_index][ccs_cfes_index]['candidate_cfes'] = len(candidate_cfes_in_subgraph)
				gcfes['stats'][subgroup_index][ccs_cfes_index]['factuals'] = len(factuals_in_subgraph)
				if verbose:
					print(f"    Nodes in connected component: {len(connected_component)}\n    Candidate Counterfactuals: {len(candidate_cfes_in_subgraph)}\n    Factuals: {len(factuals_in_subgraph)}")

				for factual in tqdm(factuals_in_subgraph, desc='Finding candidate cfes for Factuals'):
					not_possible_to_cover_factuals_group[subgroup_index][factual] = False # Initialize the Factual point as not possible to cover
					if cost_function == "max_vector_distance":
						visited = self.bfs(subgraph, factual)
						visited_candidate_cfes = visited & candidate_cfes_in_subgraph

						if visited_candidate_cfes != set():
							not_possible_to_cover_factuals_group[subgroup_index][factual] = True
							for positive_point in visited_candidate_cfes:
		
								distance = distances[factual, positive_point]
							
								if distance <= max_d:
									if positive_point not in cfes:
										cfes[positive_point] = {'Covered_recourse_points': [factual], 'Decision_based_on_this_cost': distance,
																'Distance_cost': {factual: distance}, 'Num_covered': 1, 'cc': ccs_cfes_index}
									elif factual not in cfes[positive_point]['Covered_recourse_points']:
										cfes[positive_point]['Covered_recourse_points'].append(factual)
										cfes[positive_point]['Decision_based_on_this_cost'] += distance
										cfes[positive_point]['Distance_cost'][factual] = distance
										cfes[positive_point]['Num_covered'] += 1
										cfes[positive_point]['cc'] = ccs_cfes_index

					else:
						paths = self.dijkstra_like_paths_to_positive_points(subgraph, factual, candidate_cfes_in_subgraph)
						if paths != {}:
							not_possible_to_cover_factuals_group[subgroup_index][factual] = True

						for positive_point, (path, weight) in paths.items():
							path_cost_dist = sum([subgraph[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1)])

							reachable_point = False
							decision_based_on_this_cost = 0
							if cost_function == "max_path_cost":
								if weight <= max_d:
									decision_based_on_this_cost = weight
									reachable_point = True
							elif cost_function == "num_path_hops":
								if len(path) <= max_d:
									decision_based_on_this_cost = len(path)
									reachable_point = True
							
							if reachable_point:
								if positive_point not in cfes:
									cfes[positive_point] = {'Covered_recourse_points': [factual], 'Decision_based_on_this_cost': decision_based_on_this_cost,
															'Shortest_path_cost': {factual: weight}, 'Shortest_paths_distance_cost': {factual: path_cost_dist},
															'Path': {factual: path}, 'Num_covered': 1, 'cc': ccs_cfes_index}
								elif factual not in cfes[positive_point]['Covered_recourse_points']:
									cfes[positive_point]['Covered_recourse_points'].append(factual)
									cfes[positive_point]['Decision_based_on_this_cost'] += decision_based_on_this_cost
									cfes[positive_point]['Shortest_path_cost'][factual] = weight
									cfes[positive_point]['Shortest_paths_distance_cost'][factual] = path_cost_dist
									cfes[positive_point]['Path'][factual] = path
									cfes[positive_point]['Num_covered'] += 1
									cfes[positive_point]['cc'] = ccs_cfes_index

				if k_selection_method == "accross_all_ccs":
					ccs_cfes.update(cfes)

				if k_selection_method == "same_k_for_all_ccs":
					start_time = time.time()
					if cfe_selection_method == 'greedy':
						ccs_cfes[ccs_cfes_index] = self.greedy_selection(cfes, k)
					elif cfe_selection_method =="mip":
						ccs_cfes[ccs_cfes_index] = self.mip_selection(cfes, k)
					time_fgces = time.time() - start_time


				ccs_cfes_index += 1

			if ccs_not_applicable > 0:
				if verbose:
					print(f"Total not applicable connected components for group {subgroup_index}: {ccs_not_applicable}")
					print(f"Total applicable connected components for group: {len(connected_components) - ccs_not_applicable}")
			gcfes['stats'][subgroup_index]['not_applicable_ccs'] = ccs_not_applicable

			if k_selection_method == "accross_all_ccs":
				start_time = time.time()
				if cfe_selection_method == 'greedy':
					ccs_cfes = self.greedy_selection(ccs_cfes, k)
				elif cfe_selection_method =="mip":
					ccs_cfes = self.mip_selection(ccs_cfes, k)
				time_fgces = time.time() - start_time
				if verbose:
					print(f"Number of CFEs needed: {len(ccs_cfes)}")

			gcfes[subgroup_index] = {}
			gcfes[subgroup_index] = ccs_cfes

		if total_ccs_not_applicable > 0:
			if verbose:
				print(f"Total not applicable connected components: {total_ccs_not_applicable}")
				print(f"Total applicable connected components: {total_ccs_applicable}")

		for subgroup_index in gcfes:
			if k_selection_method == "greeedy_accross_all_ccs":
				if verbose:
					print(f"Number of CFEs needed: {len(gcfes[subgroup_index])}")
			elif k_selection_method == "same_k_for_all_ccs":
				cfes_needed = sum([len(gcfes[subgroup_index][cc]) for cc in gcfes[subgroup_index]])
				if verbose:
					print(f"Number of CFEs needed: {cfes_needed}")

		return gcfes, not_possible_to_cover_factuals_group, time_fgces
	
	def apply_cfes(self, gcfes, FN_negatives_by_group, distances, not_possible_to_cover_factuals_group, k_selection_method="accross_all_ccs", cost_function="max_vector_distance", stats=None, binary_implementation=False, verbose=False):
		"""
		Applies the group CFES to the data and generates the results.

		# Parameters:
		----------
		- gcfes: (dict)
			group CFES for each subgroup
		- FN_negatives_by_group: (dict)
			factual instances by group
		- distances: (dict)
			distances between instances
		- not_possible_to_cover_factuals_group: (dict)
			factual instances that are not possible to cover
		- k_selection_method: (str)
			the method to use to select the CFES
		- cost_function: (str)
			the method to use to compute d
		- stats: (dict)
			statistics of the data
		- binary_implementation: (bool)
			whether to use the binary implementation or not
		
		# Returns:
		----------
		- results: (dict)
			results of the group CFES
		- totalcoverage: (set)
			total coverage
		- graph_stats: (dict)
			graph statistics
		- fns_div_for_both_groups: (int)
			number of factual instances
		"""
		results = {}
		totalcoverage = set()
		graph_stats = {}
		fns_div_for_both_groups = 0

		for group in gcfes:
			if group == 'stats':
				graph_stats[group] = gcfes[group]
				continue
			distances_for_group = {}
			path_cost_for_group = {}

			covered_points_for_group = set()
			results[group] = {}
			results[group]['Coverage'] = 0
			results[group]['Avg. distance'] = 0
			results[group]['Median distance'] = 0
			results[group]['Avg. path cost'] = 0
			results[group]['Median path cost'] = 0

			if k_selection_method == "accross_all_ccs":
				for cfe in gcfes[group]:
					print(f"Group {group} - CFE: {cfe} - Covered Points: {len(gcfes[group][cfe]['Covered_recourse_points'])}")
					for point in gcfes[group][cfe]['Covered_recourse_points']:
						covered_points_for_group.add(point)
						if point in results[group]:
							if cost_function == "max_vector_distance":
								dist = distances[point][cfe]
								if dist < results[group][point]['Vector_distance'] and results[group][point]['cfe_cc'] == gcfes[group][cfe]['cc']:
									results[group][point]['CFE_name'] = cfe
									dist = distances[point][cfe]
									results[group][point]['Vector_distance'] = dist
									results[group][point]['cfe_cc'] = gcfes[group][cfe]['cc']
									if (point, cfe) in distances_for_group:
										del distances_for_group[(point, cfe)]
									distances_for_group[(point, cfe)] = dist
							elif cost_function == "max_path_cost":
								if gcfes[group][cfe]['Shortest_path_cost'][point] < results[group][point]['Shortest_path_cost'] and results[group][point]['cfe_cc'] == gcfes[group][cfe]['cc']:
									results[group][point]['CFE_name'] = cfe
									path_cost = gcfes[group][cfe]['Shortest_path_cost'][point]
									results[group][point]['Shortest_path_cost'] = path_cost
									results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cfe]['Shortest_paths_distance_cost'][point]
									dist = distances[point][cfe]
									results[group][point]['Vector_distance'] = dist
									results[group][point]['cfe_cc'] = gcfes[group][cfe]['cc']
									results[group][point]['Path'] = gcfes[group][cfe]['Path'][point]
									if (point, cfe) in distances_for_group:
										del distances_for_group[(point, cfe)]
										del path_cost_for_group[(point, cfe)]
									distances_for_group[(point, cfe)] = dist
									path_cost_for_group[(point, cfe)] = path_cost
							elif cost_function == "num_path_hops":
								if len(gcfes[group][cfe]['Path'][point]) < len(results[group][point]['Path']) and results[group][point]['cfe_cc'] == gcfes[group][cfe]['cc']:
									results[group][point]['CFE_name'] = cfe
									path_cost = gcfes[group][cfe]['Shortest_path_cost'][point]
									results[group][point]['Shortest_path_cost'] = path_cost
									results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cfe]['Shortest_paths_distance_cost'][point]
									dist = distances[point][cfe]
									results[group][point]['Vector_distance'] = dist
									results[group][point]['cfe_cc'] = gcfes[group][cfe]['cc']
									results[group][point]['Path'] = gcfes[group][cfe]['Path'][point]
									if (point, cfe) in distances_for_group:
										del distances_for_group[(point, cfe)]
										del path_cost_for_group[(point, cfe)]
									distances_for_group[(point, cfe)] = dist
									path_cost_for_group[(point, cfe)] = path_cost
						else:
							results[group][point] = {}
							results[group][point]['CFE_name'] = cfe
							dist = distances[point][cfe]
							distances_for_group[(point, cfe)] = dist
							
							if cost_function != "max_vector_distance":
								path_cost = gcfes[group][cfe]['Shortest_path_cost'][point]
								path_cost_for_group[(point, cfe)] = path_cost
								results[group][point]['Shortest_path_cost'] = path_cost
								results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cfe]['Shortest_paths_distance_cost'][point]
								results[group][point]['Vector_distance'] = dist
								results[group][point]['cfe_cc'] = gcfes[group][cfe]['cc']
								results[group][point]['Path'] = gcfes[group][cfe]['Path'][point]
							
							else:
								results[group][point]['Vector_distance'] = dist
								results[group][point]['cfe_cc'] = gcfes[group][cfe]['cc']					
			elif k_selection_method == "same_k_for_all_ccs" and not binary_implementation:
				for cc in gcfes[group]:
					for cfe in gcfes[group][cc]:
						for point in gcfes[group][cc][cfe]['Covered_recourse_points']:
							covered_points_for_group.add(point)
							if point in results[group]:
								if cost_function == "max_vector_distance":
									dist = distances[point][cfe]
									if dist < results[group][point]['Vector_distance'] and results[group][point]['cfe_cc'] == gcfes[group][cfe]['cc']:
										results[group][point]['CFE_name'] = cfe
										dist = distances[point][cfe]
										results[group][point]['Vector_distance'] = dist
										results[group][point]['cfe_cc'] = gcfes[group][cc][cfe]['cc']
										if (point, cfe) in distances_for_group:
											del distances_for_group[(point, cfe)]
										distances_for_group[(point, cfe)] = dist
								elif cost_function == "max_path_cost":
									if gcfes[group][cc][cfe]['Shortest_path_cost'][point] < results[group][point]['Shortest_path_cost'] and results[group][point]['cfe_cc'] == gcfes[group][cfe]['cc']:
										results[group][point]['CFE_name'] = cfe
										path_cost = gcfes[group][cc][cfe]['Shortest_path_cost'][point]
										results[group][point]['Shortest_path_cost'] = path_cost
										results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cc][cfe]['Shortest_paths_distance_cost'][point]
										dist = distances[point][cfe]
										results[group][point]['Vector_distance'] = dist
										results[group][point]['cfe_cc'] = gcfes[group][cc][cfe]['cc']
										results[group][point]['Path'] = gcfes[group][cc][cfe]['Path'][point]
										if (point, cfe) in distances_for_group:
											del distances_for_group[(point, cfe)]
											del path_cost_for_group[(point, cfe)]
										distances_for_group[(point, cfe)] = dist
										path_cost_for_group[(point, cfe)] = path_cost
								elif cost_function == "num_path_hops":
									if len(gcfes[group][cc][cfe]['Path'][point]) < len(results[group][point]['Path']) and results[group][point]['cfe_cc'] == gcfes[group][cfe]['cc']:
										results[group][point]['CFE_name'] = cfe
										path_cost = gcfes[group][cc][cfe]['Shortest_path_cost'][point]
										results[group][point]['Shortest_path_cost'] = path_cost
										results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cc][cfe]['Shortest_paths_distance_cost'][point]
										dist = distances[point][cfe]
										results[group][point]['Vector_distance'] = dist
										results[group][point]['cfe_cc'] = gcfes[group][cc][cfe]['cc']
										results[group][point]['Path'] = gcfes[group][cc][cfe]['Path'][point]
										if (point, cfe) in distances_for_group:
											del distances_for_group[(point, cfe)]
											del path_cost_for_group[(point, cfe)]
										distances_for_group[(point, cfe)] = dist
										path_cost_for_group[(point, cfe)] = path_cost
							else:
								results[group][point] = {}
								results[group][point]['CFE_name'] = cfe
								dist = distances[point][cfe]
								distances_for_group[(point, cfe)] = dist

								if cost_function != "max_vector_distance":
									path_cost = gcfes[group][cfe]['Shortest_path_cost'][point]
									path_cost_for_group[(point, cfe)] = path_cost
									results[group][point]['Shortest_path_cost'] = path_cost
									results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cc][cfe]['Shortest_paths_distance_cost'][point]
									results[group][point]['Vector_distance'] = dist
									results[group][point]['cfe_cc'] = gcfes[group][cc][cfe]['cc']
									results[group][point]['Path'] = gcfes[group][cc][cfe]['Path'][point]
								else:
									results[group][point]['Vector_distance'] = dist
									results[group][point]['cfe_cc'] = gcfes[group][cc][cfe]['cc']
			elif k_selection_method == "same_k_for_all_ccs" and binary_implementation:
				for cc in gcfes[group]:
					if cc == 'optimal_d': 
						continue
					for key in gcfes[group][cc]:
						if key == 'optimal_d': continue
						for cfe in gcfes[group][cc][key]:
							for point in gcfes[group][cc][key][cfe]['Covered_recourse_points']:
								covered_points_for_group.add(point)
								if point in results[group]:
									if cost_function == "max_vector_distance":
										dist = distances[point][cfe]
										if dist < results[group][point]['Vector_distance'] and results[group][point]['cfe_cc'] == gcfes[group][cc][key][cfe]['cc']:
											results[group][point]['CFE_name'] = cfe
											dist = distances[point][cfe]
											results[group][point]['Vector_distance'] = dist
											results[group][point]['cfe_cc'] = gcfes[group][cc][key][cfe]['cc']
											if (point, cfe) in distances_for_group:
												del distances_for_group[(point, cfe)]
											distances_for_group[(point, cfe)] = dist
									elif cost_function == "max_path_cost":
										if gcfes[group][cc][key][cfe]['Shortest_path_cost'][point] < results[group][point]['Shortest_path_cost'] and results[group][point]['cfe_cc'] == gcfes[group][cc][key][cfe]['cc']:
											results[group][point]['CFE_name'] = cfe

											path_cost = gcfes[group][cc][key][cfe]['Shortest_path_cost'][point]
											results[group][point]['Shortest_path_cost'] = path_cost
											results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cc][key][cfe]['Shortest_paths_distance_cost'][point]
											dist = distances[point][cfe]
											results[group][point]['Vector_distance'] = dist
											results[group][point]['cfe_cc'] = gcfes[group][cc][key][cfe]['cc']
											results[group][point]['Path'] = gcfes[group][cc][key][cfe]['Path'][point]
											if (point, cfe) in distances_for_group:
												del distances_for_group[(point, cfe)]
												del path_cost_for_group[(point, cfe)]
											distances_for_group[(point, cfe)] = dist
											path_cost_for_group[(point, cfe)] = path_cost
									elif cost_function == "num_path_hops":
										if len(gcfes[group][cc][key][cfe]['Path'][point]) < len(results[group][point]['Path']) and results[group][point]['cfe_cc'] == gcfes[group][cc][key][cfe]['cc']:
											results[group][point]['CFE_name'] = cfe

											path_cost = gcfes[group][cc][key][cfe]['Shortest_path_cost'][point]
											results[group][point]['Shortest_path_cost'] = path_cost
											results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cc][key][cfe]['Shortest_paths_distance_cost'][point]
											dist = distances[point][cfe]
											results[group][point]['Vector_distance'] = dist
											results[group][point]['cfe_cc'] = gcfes[group][cc][key][cfe]['cc']
											results[group][point]['Path'] = gcfes[group][cc][key][cfe]['Path'][point]
											if (point, cfe) in distances_for_group:
												del distances_for_group[(point, cfe)]
												del path_cost_for_group[(point, cfe)]
											distances_for_group[(point, cfe)] = dist
											path_cost_for_group[(point, cfe)] = path_cost
								else:
									results[group][point] = {}
									results[group][point]['CFE_name'] = cfe
									dist = distances[point][cfe]
									distances_for_group[(point, cfe)] = dist
									
									if cost_function != "max_vector_distance":
										path_cost =gcfes[group][cc][key][cfe]['Shortest_path_cost'][point]
										path_cost_for_group[(point, cfe)] = path_cost
										results[group][point]['Shortest_path_cost'] = path_cost
										results[group][point]['Shortest_paths_distance_cost'] = gcfes[group][cc][key][cfe]['Shortest_paths_distance_cost'][point]
										results[group][point]['Vector_distance'] = dist
										results[group][point]['cfe_cc'] = gcfes[group][cc][key][cfe]['cc']
										results[group][point]['Path'] = gcfes[group][cc][key][cfe]['Path'][point]
									else:
										results[group][point]['Vector_distance'] = dist
										results[group][point]['cfe_cc'] = gcfes[group][cc][key][cfe]['cc']
		
			not_possible_to_cover = set([point for point in not_possible_to_cover_factuals_group[group] if not_possible_to_cover_factuals_group[group][point] == False])
			possible_to_cover_points = set([point for point in not_possible_to_cover_factuals_group[group] if not_possible_to_cover_factuals_group[group][point] == True])
			factuals_covered_points_for_group = set(covered_points_for_group) & possible_to_cover_points
			totalcoverage.update(factuals_covered_points_for_group)
			fns_div = len(possible_to_cover_points)
			fns_div_for_both_groups += fns_div
			if fns_div == 0:
				coverage = 0
			else:
				coverage = (len(factuals_covered_points_for_group)/fns_div)*100
			results[group]['Coverage'] = coverage

			distance_values = list(distances_for_group.values())
			avg_distance = np.mean(distance_values) if distance_values != [] else None
			median_distance = np.median(distance_values) if distance_values != [] else None
			
			results[group]['Avg. distance'] = avg_distance
			results[group]['Median distance'] = median_distance

			print(f"\n\nGroup {group} - Coverage: {coverage}%")
			if verbose:
				print(f"Group {group} - Avg. distance: {avg_distance}")
				print(f"Group {group} - Median distance: {median_distance}")

			if cost_function != "max_vector_distance":
				path_cost_values = list(path_cost_for_group.values())
				avg_path_cost = np.mean(path_cost_values) if path_cost_values != [] else None
				median_path_cost = np.median(path_cost_values) if path_cost_values != [] else None
				results[group]['Avg. path cost'] = avg_path_cost if not np.isnan(avg_path_cost) else None
				results[group]['Median path cost'] = median_path_cost if not np.isnan(median_path_cost) else None

				if verbose:
					print(f"Group {group} - Avg. path cost: {avg_path_cost}")
					print(f"Group {group} - Median path cost: {median_path_cost}")

		final_results = {}
		if stats is not None:
			final_results['Node Connectivity'] = stats['Node Connectivity']
			final_results['Edge Connectivity'] = stats['Edge Connectivity']
			if "Optimal d" in stats:
				for group in stats['Optimal d']:
					final_results[f'Optimal d - {group}'] = stats['Optimal d'][group]
		
		final_results['Total coverage'] = (len(totalcoverage)/fns_div_for_both_groups)*100 if fns_div_for_both_groups != 0 else 0
		print(f"Total coverage: {final_results['Total coverage']}%")
		final_results.update(results)

		final_results['Graph Stats'] = graph_stats['stats']

		return final_results

# =============================================================================================================
# =============================================================================================================
# ============================coverage-constrained group counterfactuals=======================================
# =============================================================================================================
# =============================================================================================================
	def get_candidate_cfes(self, G, factuals_in_subgraph, candidate_cfes_in_subgraph, max_d, cost_function, distances, ccs_cfes_index):
		"""
		Returns a dictionary of candidate cfes for a given subgroup.

		# Arguments:
		----------
		- G: (networkx.Graph)
			subgraph
		- factuals_in_subgraph: (list)
			factual instances
		- candidate_cfes_in_subgraph: (list)	
			positive class instances
		- max_d: (float)
			maximum value of d
		- cost_function: (str)
			method to compute d
		- distances: (dict)
			distances between instances
		- ccs_cfes_index: (int)
			index of the ccs cfes
		
		# Returns:
		----------
		- cfes: (dict)
			candidate cfes
		- not_possible_to_cover_factuals: (set)
			factual instances that are not possible to cover 
		"""
		not_possible_to_cover_factuals = {}
		cfes = {}

		for factual in factuals_in_subgraph:
			not_possible_to_cover_factuals[factual] = False

			if cost_function == "max_vector_distance":
				visited = self.bfs(G, factual)
				visited_candidate_cfes = visited & candidate_cfes_in_subgraph

				if visited_candidate_cfes != set():
					not_possible_to_cover_factuals[factual] = True
					for positive_point in visited_candidate_cfes:
						distance = distances[factual, positive_point]
						if distance <= max_d:
							cfes.setdefault(positive_point, {'Covered_recourse_points': [], 'Decision_based_on_this_cost': 0,
																'Distance_cost': {}, 'Num_covered': 0})
							cfes[positive_point]['Covered_recourse_points'].append(factual)
							cfes[positive_point]['Decision_based_on_this_cost'] += distance
							cfes[positive_point]['Distance_cost'][factual] = distance
							cfes[positive_point]['Num_covered'] += 1
							cfes[positive_point]['cc'] = ccs_cfes_index
			else:
				paths = self.dijkstra_like_paths_to_positive_points(G, factual, candidate_cfes_in_subgraph)
				if paths != {}:
					not_possible_to_cover_factuals[factual] = True

				for positive_point, (path, weight) in paths.items():
					path_cost_dist = sum([G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1)])

					reachable_point = False
					decision_based_on_this_cost = 0
					if cost_function == "max_path_cost":
						if weight <= max_d:
							decision_based_on_this_cost = weight
							reachable_point = True
					elif cost_function == "num_path_hops":
						if len(path) <= max_d:
							decision_based_on_this_cost = len(path)
							reachable_point = True
					
					if reachable_point:
						cfes.setdefault(positive_point, {'Covered_recourse_points': [], 'Decision_based_on_this_cost': 0,
														'Shortest_path_cost': {}, 'Shortest_paths_distance_cost': {}, "Path": {}, 'Num_covered': 0})
						cfes[positive_point]['Covered_recourse_points'].append(factual)
						cfes[positive_point]['Decision_based_on_this_cost'] += decision_based_on_this_cost
						cfes[positive_point]['Shortest_path_cost'][factual] = weight
						cfes[positive_point]['Shortest_paths_distance_cost'][factual] = path_cost_dist
						cfes[positive_point]['Path'][factual] = path
						cfes[positive_point]['Num_covered'] += 1
						cfes[positive_point]['cc'] = ccs_cfes_index
		return cfes, not_possible_to_cover_factuals

	def binary_search_for_d(self, G, factuals, candidate_cfes, k, min_d, max_d, bst, cost_function, distances, ccs_cfes_index, find_k0=True):
		"""
		Performs a binary search for the optimal value of d.

		# Arguments:
		----------
		- G: (networkx.Graph)
			subgraph
		- factuals: (list)
			factual instances
		- candidate_cfes: (list)
			positive class instances
		- k: (int)
			number of cfes to be returned
		- min_d: (float)
			minimum value of d
		- max_d: (float)
			maximum value of d
		- bst: (float)
			minimum value of d for the binary search
		- cost_function: (str)
			cost function to use
		- distances: (dict)
			distances between instances
		- ccs_cfes_index: (int)
			index of the ccs cfes
		
		# Returns:
		----------
		- optimal_d: (float)
			optimal value of d
		- selected_cfes: (dict)
			selected cfes
		"""
		optimal_d = max_d
		optimal_coverage = -1
		not_possible_to_cover_factuals = {}
		selected_cfes = {}
		min_selected_cfes = float('inf')

		while max_d - min_d > bst:
			mid_d = (min_d + max_d) / 2

			cfes, notc = self.get_candidate_cfes(G, factuals, candidate_cfes, mid_d, cost_function, distances, ccs_cfes_index)
			current_selected_cfes = self.greedy_selection(cfes, k)

			covered_negatives = set()
			for cfe_index, _ in current_selected_cfes.items():
				covered_negatives.update(current_selected_cfes[cfe_index]['Covered_recourse_points'])

			uncovered_negatives = set(factuals) - covered_negatives
			not_possible_to_cover_fns_for_state = set([point for point in notc if notc[point] == False])
			
			current_coverage = len(factuals) - len(uncovered_negatives)
			if current_coverage > optimal_coverage or (current_coverage == optimal_coverage and mid_d < optimal_d):
				if min_selected_cfes >= len(current_selected_cfes) and find_k0:
					min_selected_cfes = len(current_selected_cfes)
					k = min_selected_cfes
					optimal_coverage = current_coverage
					optimal_d = mid_d
					not_possible_to_cover_factuals = notc
					selected_cfes = current_selected_cfes
				else:
					optimal_coverage = current_coverage
					optimal_d = mid_d
					not_possible_to_cover_factuals = notc
					selected_cfes = current_selected_cfes
		
			possible_to_cover_points = set(factuals) - not_possible_to_cover_fns_for_state
			if possible_to_cover_points == covered_negatives:
				max_d = mid_d
			else:
				min_d = mid_d
		print(f"			# {len(selected_cfes)} CFEs for the {ccs_cfes_index} cc")

		return optimal_d, selected_cfes, not_possible_to_cover_factuals
	
	def compute_gcfes_binary(self, subgroups, candidate_cfes, factuals, k, min_d, max_d, bst, cost_function, distances, find_k0):
		"""
		Computes the group CFES for each subgroup.

		# Parameters:
		----------
		- subgroups: (dict)
			Subgroups of the graph
		- candidate_cfes: (list)
			Positive class instances
		- factuals: (list)
			Instances under explanation
		- k: (int)
			The maximum number of CFES to return
		- min_d: (float)
			The minimum value of d
		- max_d: (float)
			The maximum value of d
		- bst: (float)
			The minimum value of d for the binary search
		- cost_function: (str)
			The method to use to compute d
		- distances: (dict)
			Distances between instances
		
		# Returns:
		----------
		- gcfes: (dict)
			Group CFES for each subgroup
		- gcfes_with_ccs: (dict)
			Group CFES for each subgroup with connected components
		- not_possible_to_cover_factuals_group: (dict)
			Instances under explanation that are not possible to cover
		"""
		gcfes = {}
		gcfes['stats'] = {}
		gcfes_with_ccs = {}
		total_ccs_not_applicable = 0
		not_possible_to_cover_factuals_group = {}

		ccs_cfes_index = 0
		
		for subgroup_index, subgroup in subgroups.items():
			connected_components = list(nx.weakly_connected_components(subgroup))
			gcfes[subgroup_index] = {}
			gcfes_with_ccs[subgroup_index] = {}
			not_possible_to_cover_factuals_group[subgroup_index] = {}

			gcfes['stats'][subgroup_index] = {}
			gcfes['stats'][subgroup_index]['nodes'] = len(subgroup)
			gcfes['stats'][subgroup_index]['connected_components'] = len(connected_components)
			print(f"Subgroup: {subgroup_index} - Nodes: {len(subgroup)}\nConnected components: {len(connected_components)}")

			ccs_cfes = {}
			ccs_not_applicable = 0
			
			for connected_component in connected_components:
				subgraph = subgroup.subgraph(connected_component)
			
				subgraph = subgroup.subgraph(connected_component)
				candidate_cfes_in_subgraph = set(connected_component) & set(candidate_cfes)
				factuals_in_subgraph = set(connected_component) & set(factuals)

				if not factuals_in_subgraph or not candidate_cfes_in_subgraph:
					ccs_not_applicable += 1
					total_ccs_not_applicable += 1
					ccs_cfes_index += 1
					continue
			
				gcfes['stats'][subgroup_index][ccs_cfes_index] = {}
				gcfes['stats'][subgroup_index][ccs_cfes_index]['nodes'] = len(connected_component)
				gcfes['stats'][subgroup_index][ccs_cfes_index]['candidate_cfes'] = len(candidate_cfes_in_subgraph)
				gcfes['stats'][subgroup_index][ccs_cfes_index]['factuals'] = len(factuals_in_subgraph)
				print(f"    Nodes in connected component: {len(connected_component)}\n    Candidate cfes: {len(candidate_cfes_in_subgraph)}\n    Factuals: {len(factuals_in_subgraph)}")

				optimal_d, selected_cfes, not_possible_to_cover_factuals = self.binary_search_for_d(subgraph, factuals_in_subgraph, candidate_cfes_in_subgraph, k, min_d, max_d, bst, cost_function, distances, ccs_cfes_index, find_k0)
				print(f"	Optimal d for subgraph: {optimal_d}")

				for point, value in not_possible_to_cover_factuals.items():
					not_possible_to_cover_factuals_group[subgroup_index][point] = value

				ccs_cfes[ccs_cfes_index] = {}
				ccs_cfes[ccs_cfes_index]['optimal_d'] = optimal_d
				ccs_cfes[ccs_cfes_index]['selected_cfes'] = {}
				if selected_cfes:
					ccs_cfes[ccs_cfes_index]['selected_cfes'] = selected_cfes
				ccs_cfes_index += 1

			gcfes[subgroup_index] = ccs_cfes
			
			if ccs_not_applicable > 0:
				print(f"	Total not applicable connected components for group {subgroup_index}: {ccs_not_applicable}")
				print(f"	Total applicable connected components for group {subgroup_index}: {len(connected_components) - ccs_not_applicable}")

			gcfes_with_ccs[subgroup_index] = ccs_cfes

			cfes_needed = sum([len(gcfes[subgroup_index][cc]['selected_cfes']) for cc in gcfes[subgroup_index]])
			print(f"	Number of CFEs needed for group {subgroup_index}: {cfes_needed}\n")

		if total_ccs_not_applicable > 0:
			print(f"Total not applicable connected components: {total_ccs_not_applicable}")

		return gcfes, gcfes_with_ccs, not_possible_to_cover_factuals_group

# ====================================================================================================
# ====================================================================================================
# ============================coverage-constrained group counterfactuals MIPS=========================
# ====================================================================================================
# ====================================================================================================
	def preprocess_connected_components(self, subgroup, connected_components, factuals, candidate_cfes):
		"""
		Preprocesses connected components to filter out those without both factuals and candidate counterfactuals that can be used.

		# Parameters:
		----------
		- subgroup: (networkx.Graph)
			Subgroup graph.
		- connected_components: (list) 
			List of connected components.
		- factuals: (list)
			List of indices of factuals.
		- candidate_cfes: (list)
			List of indices of positive points.

		# Returns:
		- connected_components_to_use: (list)
			List of connected components to use, filtered to contain only those with both factuals and positives.
		- node_to_cc: (dict)
			Mapping from node index to connected component index.
		- factuals_with_path_per_group: (list)
			List of factuals with path per group.
		- positive_points_with_path_per_group: (list)
			List of positive points with path per group.
		- factuals_positives_per_cc: (dict)
			Mapping from connected component index to lists of factuals and positive points.
		- positives_per_negative: (dict)
			Mapping from factual point to associated cfes.
		"""
		connected_components_to_use = []
		node_to_cc = {}
		factuals_with_path_per_group = []
		positive_points_with_path_per_group = []
		factuals_positives_per_cc = {}
		positives_per_negative = {}
		index=0

		for connected_component in connected_components:
			positive_points_with_path = set()
			factuals_with_path = set()
			subgraph = subgroup.subgraph(connected_component)
			
			candidate_cfes_in_subgraph = set(connected_component) & set(candidate_cfes)
			factuals_in_subgraph = set(connected_component) & set(factuals)
			
			if factuals_in_subgraph and candidate_cfes_in_subgraph:
				for fn in factuals_in_subgraph:	
					visited_nodes = self.bfs(subgraph, fn)
					if len(visited_nodes) > 0:
						pos = set(visited_nodes) & set(candidate_cfes)
						if len(pos) > 0:			 
							factuals_with_path.add(fn)
							positive_points_with_path.update(pos)
							positives_per_negative[fn] = pos

			if len(factuals_with_path) > 0 and len(positive_points_with_path) > 0:
				connected_components_to_use.append(connected_component)
				print(f"Factuals with path to at least one positive point = {len(factuals_with_path)}")
				print(f"Positive points with path to at least one factual = {len(positive_points_with_path)}")
				factuals_with_path_lst = list(factuals_with_path)
				positive_points_with_path_lst = list(positive_points_with_path)
				factuals_with_path_per_group += factuals_with_path_lst
				positive_points_with_path_per_group += positive_points_with_path_lst
				all_points = factuals_with_path_lst + positive_points_with_path_lst
				node_to_cc.update({i: index for i in all_points})
				factuals_positives_per_cc[index] = [factuals_with_path_lst,positive_points_with_path_lst]
				index+=1

		return connected_components_to_use, node_to_cc, factuals_with_path_per_group, positive_points_with_path_per_group, factuals_positives_per_cc,positives_per_negative
	
	def get_gcfes_approach_integer_prog_local(self, subgroups, distances, candidate_cfes, factuals):
		"""
		Perform the GCFES approach using integer programming locally per connected component.

		# Parameters:
		- subgroups: (dict)
			Dictionary of subgroups represented as networkx.Graph objects.
		- distances: (dict)
			Dictionary containing distances between points.
		- candidate_cfes: (list)
			List of indices of positive points.
		- factuals: (list)
			List of indices of instances under explanation.

		# Returns:
		- gcfes: (dict)
			Dictionary of GCFES (global counterfactual explanations) per subgroup.
		- valid_connected_components_per_subgroup: (dict) 
			Dictionary of valid connected components per subgroup.
		- k_per_component_per_group: (dict) 
			Dictionary of appropriate k per component per subgroup to achieve full coverage.
		- results: (dict) 
			Dictionary containing max cost per subgroup.
		- candidate_cfes_per_factuals_per_group: (dict)
			Dictionary containing positive points per factual point per subgroup.
		- factuals_positives_per_cc: (dict)
			Dictionary containing factuals-positive pairs.
		"""
		results = {}
		k_per_component_per_group = {}
		valid_connected_components_per_subgroup = {}
		gcfes = {}
		candidate_cfes_per_factuals_per_group = {}
		factual_positive_per_cc_for_global = {}

		for subgroup_index, subgroup in subgroups.items():
			max_cost_over_ccs = 0
			total_coverage_per_group = 0
			connected_components = []
			gcfes[subgroup_index] = {}
			print(f"----------------------------------------------")
			print(f"Group = {subgroup_index}")
			results_per_connected_component = {}
			connected_components = list(nx.weakly_connected_components(subgroup))
			
			 
			connected_components_to_use, _, _, _, factuals_positives_per_cc, positives_per_negative = self.preprocess_connected_components(subgroup, connected_components, factuals, candidate_cfes)
			factual_positive_per_cc_for_global[subgroup_index] = factuals_positives_per_cc

			candidate_cfes_per_factuals_per_group[subgroup_index] = positives_per_negative
			valid_connected_components_per_subgroup[subgroup_index] = connected_components_to_use
			k_per_component = {component: 1 for component in range(len(connected_components_to_use))}

			for index, _ in enumerate(connected_components_to_use):
				results_per_connected_component[index] = {'Coverage': 0, 'Cost': 0}
			
			for idx, connected_component in enumerate(connected_components_to_use):
				k = 0
				coverage_percent = 0
				while coverage_percent < 1.0:
					k += 1
					candidate_cfes_in_subgraph = set(connected_component) & set(candidate_cfes)
					factuals_in_subgraph = set(connected_component) & set(factuals)
					if factuals_in_subgraph and candidate_cfes_in_subgraph:
						factuals_with_path = factuals_positives_per_cc[idx][0]
						positive_points_with_path = factuals_positives_per_cc[idx][1]
						vector_distances = []
						counter = 0
						for factual in factuals_with_path:
							positive_points_connected = positives_per_negative[factual]
							dist_per_factual = []
							for positive_point in positive_points_with_path:
								if positive_point in positive_points_connected:
									counter += 1
									dist = distances[factual][positive_point]
									dist_per_factual.append(dist)
								else:
									dist_per_factual.append(100)
							vector_distances.append(dist_per_factual)

						gcfes_cc, max_cost, _, coverage = self.select_cfes_mixed_integer_programming(factuals_with_path, positive_points_with_path, vector_distances, k, 1)
						if max_cost > max_cost_over_ccs:
							max_cost_over_ccs = max_cost

						gcfes[subgroup_index].update(gcfes_cc)
						coverage_percent = coverage / len(factuals_with_path)
						results_per_connected_component[idx]["Coverage"] = coverage_percent
						results_per_connected_component[idx]["Cost"] = max_cost
						k_per_component[idx] = k
						print(f"Coverage = {coverage_percent} with {k} cfes")
						
				k_per_component_per_group[subgroup_index] = k_per_component
			component_with_max_cost = max(results_per_connected_component, key=lambda k: results_per_connected_component[k]['Cost'])
			total_coverage_per_group = sum(value['Coverage'] for value in results_per_connected_component.values())
			results[subgroup_index] = {'Total Coverage': total_coverage_per_group, 'Total Cost': results_per_connected_component[component_with_max_cost]["Cost"], 'Components': results_per_connected_component}
			print(f"Group {subgroup_index}: K per component for coverage = {k_per_component}. Max cost = {max_cost_over_ccs}. Total k used = {sum(value for value in k_per_component.values())}")
		
		return gcfes, results, valid_connected_components_per_subgroup, k_per_component_per_group, candidate_cfes_per_factuals_per_group, factual_positive_per_cc_for_global
					
	def get_gcfes_approach_integer_prog_global_via_local(self, subgroups, distances, candidate_cfes, factuals, k, cost_function):
		"""
		Perform the GCFES approach using integer programming globally by leveraging the local approach for full coverage.
		Computes the group CFES using integer programming approach for each subgroup sharing k according to the maximum cost across connencted components.

		# Parameters:
		- subgroups: (dict)
			Dictionary containing subgroups as keys and their corresponding graphs as values.
		- distances: (dict)
			Dictionary containing pairwise distances between points in the dataset.
		- candidate_cfes: (list)
			List of indices of positive points.
		- factuals: (list)
			List of indices of factuals.
		- k: (int)
			Maximum number of cfe points

		# Returns:
		- gcfes: (dict)
			Dictionary of GCFES (global counterfactual explanations) per subgroup.
		- results: (dict) 
			max distance per subgroup and per conencted component.
		"""
		 
		results = {}
		gcfes = {}
		gcfes, results, valid_connected_components_per_subgroup, k_for_coverage, candidate_cfes_per_factuals_per_group, factual_positive_per_cc_for_global = self.get_gcfes_approach_integer_prog_local(subgroups, distances, candidate_cfes, factuals, cost_function)


		for subgroup_index, _ in subgroups.items():
			print(f"\Group {subgroup_index}")
			results_per_connected_component = {}
			gcfes[subgroup_index] = {}

			total_coverage_per_group = 0
			connected_components_to_use = valid_connected_components_per_subgroup[subgroup_index]
			k_per_component = k_for_coverage[subgroup_index]

			for index, _ in enumerate(connected_components_to_use):
				results_per_connected_component[index] = {'Coverage': 0, 'Cost': 0}
			
			k_current = sum(k_per_component.values())
			
			if len(connected_components_to_use) == 1:
				coverage_percent = 0
				connected_component = connected_components_to_use[0]
			
				candidate_cfes_in_subgraph = set(connected_component) & set(candidate_cfes)
				factuals_in_subgraph = set(connected_component) & set(factuals)

				if factuals_in_subgraph and candidate_cfes_in_subgraph:
					factuals_with_path = factual_positive_per_cc_for_global[subgroup_index][0][0]
					positive_points_with_path = factual_positive_per_cc_for_global[subgroup_index][0][1]
					vector_distances = []
					
					for factual in factuals_with_path:
						dist_per_factual = []
						positive_points_connected = set(candidate_cfes_per_factuals_per_group[subgroup_index][factual])
						for positive_point in positive_points_with_path:						
							if positive_point in positive_points_connected:
								dist = distances[factual][positive_point]
								dist_per_factual.append(dist)
							else:
								dist_per_factual.append(100)	
						vector_distances.append(dist_per_factual)	

					gcfes, max_cost, _, coverage = self.select_cfes_mixed_integer_programming(factuals_with_path, positive_points_with_path, vector_distances, k, 1)
					coverage_percent = coverage/ len(factuals_with_path)
					
					
					results_per_connected_component[0]["Coverage"] = coverage_percent
					results_per_connected_component[0]["Cost"] = max_cost
					
					component_with_max_cost = max(results_per_connected_component, key=lambda k: results_per_connected_component[k]['Cost'])	
			else:
				while k_current <= k:		
					for idx, connected_component in enumerate(connected_components_to_use):
						candidate_cfes_in_subgraph = set(connected_component) & set(candidate_cfes)
						factuals_in_subgraph = set(connected_component) & set(factuals)

						if factuals_in_subgraph and candidate_cfes_in_subgraph:
							factuals_with_path = factual_positive_per_cc_for_global[subgroup_index][idx][0]
							positive_points_with_path = factual_positive_per_cc_for_global[subgroup_index][idx][1]				
							vector_distances = []

							for factual in factuals_with_path:
								dist_per_factual = []
								positive_points_connected = set(candidate_cfes_per_factuals_per_group[subgroup_index][factual])
								for positive_point in positive_points_with_path:
									if positive_point in positive_points_connected:
										dist = distances[factual][positive_point]
										dist_per_factual.append(dist)
									else:
										dist_per_factual.append(100)	
								vector_distances.append(dist_per_factual)		
						
							gcfes_cc, max_cost, _, coverage = self.select_cfes_mixed_integer_programming(factuals_with_path, positive_points_with_path, vector_distances, k_per_component[idx], 1)
							gcfes[subgroup_index].update(gcfes_cc)
							results_per_connected_component[idx]["Cost"] = max_cost
							results_per_connected_component[idx]["Coverage"] = coverage/ len(factuals_with_path)
					components_not_covered = [key for key, value in results_per_connected_component.items() if value['Coverage'] < 1.0]

					if len(components_not_covered) > 0:
						for component in components_not_covered:
							k_per_component[component] += 1		
						component_with_max_cost = max(results_per_connected_component, key=lambda k: results_per_connected_component[k]['Cost'])

					else:
						component_with_max_cost = max(results_per_connected_component, key=lambda k: results_per_connected_component[k]['Cost'])
						if k_current == k:
							break	
						k_current +=1
						k_per_component[component_with_max_cost] += 1
					
			total_coverage_per_group = sum(value['Coverage'] for value in results_per_connected_component.values())
			results[subgroup_index] = {'Total Coverage': total_coverage_per_group, 'Total Cost':results_per_connected_component[component_with_max_cost]["Cost"],'Components':results_per_connected_component}	
			max_cost_value = max(results_per_connected_component.values(), key=lambda x: x['Cost'])['Cost']

			print(f"Group {subgroup_index}: K per component for coverage = {k_per_component}. Max cost = {max_cost_value}")

		return 	gcfes, results

	def get_gcfes_approach_integer_prog_global(self, subgroups, distances, candidate_cfes, factuals, k, coverage_percentage):
		"""
		Perform the GCFES approach using integer programming globally per subgroup.

		# Parameters:
		----------
		- subgroups: (dict)
			Dictionary containing subgroups as keys and their corresponding graphs as values.
		- distances: (dict) 
			Dictionary containing pairwise distances between points in the dataset.
		- candidate_cfes: (list)
			List of indices of positive points.
		- factuals: (list) 
			List of indices of factuals.
		- k: (int)
			Maximum number of cfe points to return
		- coverage_percentage: (float) 
			Percentage of points that will be covered
		
		# Returns:
			results: (dict)
				group cfes and max distance per subgroup.
		"""
		results = {}
		gcfes = {}
		idx = 0.0	
		for _, subgroup in subgroups.items():
			print(f"Group {idx}")
			connected_components = list(nx.weakly_connected_components(subgroup))
			_, node_to_cc, factuals_with_path_per_group, positive_points_with_path_per_group, _, positives_per_negative = self.preprocess_connected_components(subgroup, connected_components, factuals, candidate_cfes)
			gcfes[idx] = {}
			subgroup_nodes = list(subgroup.nodes())
			candidate_cfes_in_subgraph = set(subgroup_nodes) & set(candidate_cfes)
			factuals_in_subgraph = set(subgroup_nodes) & set(factuals)
			if factuals_in_subgraph and candidate_cfes_in_subgraph:
				factuals_with_path = factuals_with_path_per_group
				positive_points_with_path = positive_points_with_path_per_group		
				vector_distances = []
				for factual in factuals_with_path:
					dist_per_factual = []
					positive_points_connected = positives_per_negative[factual]
					for positive_point in positive_points_with_path:
						if node_to_cc[factual] == node_to_cc[positive_point]:
							if positive_point in positive_points_connected:
								dist = distances[factual][positive_point]
								dist_per_factual.append(dist)
							else:
								dist_per_factual.append(100)	
						else:
							dist_per_factual.append(100)	
					vector_distances.append(dist_per_factual)		
				gcfes_cc, max_cost, _, coverage = self.select_cfes_mixed_integer_programming(factuals_with_path, positive_points_with_path, vector_distances, k, coverage_percentage)
				gcfes[idx].update(gcfes_cc)
			results[idx] = {'Total Coverage': coverage/ len(factuals_with_path), 'Total Cost':max_cost}
			print(f"Group {idx}: CFEs used = {k}. Max cost = {max_cost}. Coverage = {coverage/len(factuals_with_path)}")	
			idx += 1
		return 	gcfes, results
	
	def select_cfes_mixed_integer_programming(self, factuals, positives, cost, k, coverage_percentage):
		"""
		Selects the CFES using mixed integer programming

		# Parameters:
		----------
		- factuals: (list) 
			list of factuals
		- positives: (list) 
			List of positive points
		- distances: (dict)
			Dictionary with distances among factuals and positives
		- k (dict): 
			Maximum number of cfe points
		- coverage_percentage: (float)
			Percentage of points that will be covered

		# Returns:
		- max_dist: (float)
			Maximum distance of negative to positive point
    	- min_dist: (float)
			Minimum distance of negative to positive point
		- coverage: (int)
			Number of negative points that are covered
		"""
		l = 100
		m = len(factuals) 
		n = len(positives) 
		prob = LpProblem("p-center", LpMinimize)
		x = LpVariable.dicts("x", [(i,j) for i in range(m) for j in range(n) if cost[i][j] != l], 0, 1, LpBinary)
		y = LpVariable.dicts("y", [j for j in range(n)], 0, 1, LpBinary) 
		d = LpVariable("d", lowBound = 0, cat='Continuous')
		prob += d
		for i in range(m):
			constraint = lpSum([x[(i,j)] for j in range(n) if cost[i][j] != l]) <= 1
			prob += constraint
		for i in range(m):
			for j in range(n):
				if cost[i][j] != l:
					constraint = x[(i,j)] <= y[j]
					prob += constraint

		constraint = lpSum([x[(i,j)] for j in range(n) for i in range(m) if cost[i][j] != l]) >= coverage_percentage*m
		prob += constraint
		constraint = lpSum([y[j] for j in range(n)]) <= k
		prob += constraint	

		for i in range(m):
			constraint = (lpSum(cost[i][j] * x[i, j] for j in range(n) if cost[i][j] != l) <= d)
			prob += constraint
		status = prob.solve(PULP_CBC_CMD(msg = False))
		max_dist = 0
		min_dist = inf
		coverage = 0
		gcfes = {}
		if LpStatus[status] == 'Optimal':
			print(f"Objective Value (d): {d.varValue}")
			for j in range(n):
				if y[j].value() == 1:
					print("cfe", positives[j], "is located.")
					for i in range(m):
						if cost[i][j] != l :					 
							if x[(i,j)].value() > 0.5:
								gcfes[factuals[i]] = positives[j]
								dist = cost[i][j]
								print(f"- factuals {factuals[i]} is covered with cost = {dist}")
								coverage += 1
								if dist > max_dist:
									max_dist = dist
								if dist < min_dist:
									min_dist = dist
		else:
			print("\nOptimal solution not found. Consider to try with a larger value of k.\n")
		return gcfes, max_dist, min_dist, coverage
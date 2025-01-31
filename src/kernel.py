import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from FGCE import *

EPSILON = 1e-8

class Kernel:
	def __init__(self, dataset_Name, data, skip_bandwith_calculation=True, bandwith_approch="optimal"):
		bandwidth = self.get_bandwith(dataset_Name, data, skip_bandwith_calculation=skip_bandwith_calculation, bandwith_approch=bandwith_approch)
		print(f"Bandwidth: {bandwidth}")
		self._kernel = KernelDensity(bandwidth=bandwidth, kernel='gaussian')

	def fitKernel(self, X):
		print("Fitting kernel...")
		self._kernel.fit(X)

	def kernelKDE(self, xi, xj, dist):
		"""
		KDE kernel values
		"""
		density_at_mean = np.exp(self._kernel.score_samples([0.5*(xi + xj)]))
		return (1/(density_at_mean + EPSILON))*dist
	
	def kernelKDEdata(self, X, distances):
		indices = list(X.keys())
		densities = np.exp(self._kernel.score_samples(list(X.values())))
		return {pair: (1/(density + EPSILON)) * distances[pair] for pair, density in zip(indices, densities)}
	
	def scotts_rule_bandwidth(self, data):
		"""
		Calculate bandwidth using Scott's rule for KDE.
		
		Parameters:
		data (array-like): Input data.
		
		Returns:
		float: Bandwidth calculated using Scott's rule.
		"""
		n = len(data)
		std_dev = np.std(data)
		bandwidth = (4 * std_dev ** 5 / (3 * n)) ** (1 / 5)
		return bandwidth
	
	def optimize_bandwidth(self, data, all_features):
		"""
		Optimize bandwidth for a given feature using GridSearchCV and KDE.
		
		Parameters:
		data (pd.Series): Input data for one feature.
		all_features (bool): If True, the input data is a DataFrame with multiple features. Else, the input data is a Series with a single feature.
		
		Returns:
		float: Optimal bandwidth for KDE.
		"""
		if all_features:
			data_reshaped = data
		else:
			data_reshaped = data.reshape(-1, 1)
		bandwidths = np.linspace(0.001, 0.1, 30)
		grid = GridSearchCV(KernelDensity(kernel='gaussian'),
							{'bandwidth': bandwidths},
							cv=LeaveOneOut())
		grid.fit(data_reshaped)
		return grid.best_params_['bandwidth']

	def get_bandwith(self, dataset_Name, data, skip_bandwith_calculation, bandwith_approch="optimal"):
		"""
		Get the bandwidth for the KDE.
		
		# Parameters:
		- dataset_Name (str): Name of the dataset.
		- data (pd.DataFrame): Input data.
		- skip_bandwith_calculation (bool): If True, skip the bandwidth calculation and use a precomputed value. These values are based on our preprocessing and are dataset-specific.
		- bandwith_approch (str): Bandwidth calculation approach. Options: 'optimal', 'mean_optimal', 'mean_scotts_rule', None.

		# Returns:
		- float: Bandwidth for the KDE.

		"""
		print(f"Get bandwidth for {dataset_Name}...")
		if dataset_Name == "Compas":
			bandwith = 0
			if bandwith_approch == "optimal":
				if skip_bandwith_calculation:
					bandwith = 0.01806896551724138
				else:
					bandwith = self.optimize_bandwidth(data, all_features=True)
			elif bandwith_approch == "mean_optimal":
				optimal_bandwidths = {}
				for column in data.columns:
					optimal_bandwidths[column] = self.optimize_bandwidth(data[column], all_features=False)
				bandwith = np.mean(list(optimal_bandwidths.values()))
			elif bandwith_approch == "mean_scotts_rule":
				if skip_bandwith_calculation:
					bandwith = 0.05217038248250577
				else:
					bandwith = self.scotts_rule_bandwidth(data).mean()
			else:
				bandwith = 0.5
			return bandwith

		elif dataset_Name == "Student":
			bandwith = 0
			if bandwith_approch == "optimal":
				if skip_bandwith_calculation:
					bandwith = 0.09317241379310345
				else:
					bandwith = self.optimize_bandwidth(data, all_features=True)
			elif bandwith_approch == "mean_optimal":
				optimal_bandwidths = {}
				for column in data.columns:
					optimal_bandwidths[column] = self.optimize_bandwidth(data[column], all_features=False)
				bandwith = np.mean(list(optimal_bandwidths.values()))
			elif bandwith_approch == "mean_scotts_rule":
				if skip_bandwith_calculation:
					bandwith = 0.10063027087295327
				else:
					print("Calculating bandwidth using Scott's rule for all features...")
					bandwith = self.scotts_rule_bandwidth(data).mean()
			return bandwith
		
		elif dataset_Name == "Heloc":
			bandwith = 0
			if bandwith_approch == "optimal":
				if skip_bandwith_calculation:
					bandwith = 0.05220689655172414
				else:
					bandwith = self.optimize_bandwidth(data, all_features=True)
			elif bandwith_approch == "mean_optimal":
				optimal_bandwidths = {}
				for column in data.columns:
					optimal_bandwidths[column] = self.optimize_bandwidth(data[column], all_features=False)
				bandwith = np.mean(list(optimal_bandwidths.values()))
			elif bandwith_approch == "mean_scotts_rule":
				if skip_bandwith_calculation:
					bandwith = 0.06888248705149994
				else:
					bandwith = self.scotts_rule_bandwidth(data).mean()
			return bandwith
		
		elif dataset_Name == "Adult":
			bandwith = 0
			if bandwith_approch == "optimal":
				bandwith = self.optimize_bandwidth(data, all_features=True)
			elif bandwith_approch == "mean_optimal":
				optimal_bandwidths = {}
				for column in data.columns:
					optimal_bandwidths[column] = self.optimize_bandwidth(data[column], all_features=False)
				bandwith = np.mean(list(optimal_bandwidths.values()))
			elif bandwith_approch == "mean_scotts_rule":
				if skip_bandwith_calculation:
					bandwith = 0.04465720232058467
				else:
					bandwith = self.scotts_rule_bandwidth(data).mean()
			return bandwith

		elif dataset_Name == "GermanCredit":
			bandwith = 0
			if bandwith_approch == "optimal":
				if skip_bandwith_calculation:
					bandwith = 0.09658620689655173
				else:
					bandwith = self.optimize_bandwidth(data, all_features=True)
			elif bandwith_approch == "mean_optimal":
				optimal_bandwidths = {}
				for column in data.columns:
					optimal_bandwidths[column] = self.optimize_bandwidth(data[column], all_features=False)
				bandwith = np.mean(list(optimal_bandwidths.values()))
			elif bandwith_approch == "mean_scotts_rule":
				if skip_bandwith_calculation:
					bandwith = 0.10679064225661136
				else:
					bandwith = self.scotts_rule_bandwidth(data).mean()
			return bandwith	
		else:
			"""
			For other datasets, return a default value. To be updated based on needs.
			"""
			return 0.5
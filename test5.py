import pandas as pd
from utils import draw_graphs
from intermediate import PCA_fit
from datetime import datetime
from analysis.datadropper2 import drop_days_after_events
from analysis import fitting_montecarlo, fitting2
import numpy as np

df = pd.read_pickle(r'output\intermediate.pkl')

station_name = 'PHKT'

df, eigs = PCA_fit.compute_and_apply_pca_ne_station(df, station_name, max_day = 365)#(datetime(2005,3,28) - datetime(2004,12,26)).days

draw_graphs.plot_column_for_station(df, station_name, 'pc1')

dayssince2012eq = (datetime(2012, 4, 11) - datetime(2004,12,26)).days
decades_since_2012eq = dayssince2012eq / 365.0 / 10.0
dayssince2005eq = (datetime(2005,3,28) - datetime(2004,12,26)).days
decades_since_2005eq = dayssince2005eq / 365.0 / 10.0


eq_dates_decades = []


df = drop_days_after_events(df,[ dayssince2012eq], 30*9)

x, y, sigma = fitting_montecarlo.prepare_monte_carlo_inputs(df, station_name, 'pc1', -365*20, 365*20)

results = fitting_montecarlo.run_monte_carlo_fit(
	x, y, sigma
	,fit_func=fitting2.double_exponential_with_earthquakes_and_linear_region
	,fit_jacob=fitting2.jacobian_double_exponential_with_earthquakes_and_linear_region
	,initial_guess=[1,1,1,1,1,1]# if doing stuff with earthquakes, intial guess should reflect number of parameters to be optimized, if it still breaks consider giving bounds
	,bounds = ([0,0,0,0,-np.inf,0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, ])
	,n_simulations=1000
	,n_jobs=20
	, earthquake_dates=eq_dates_decades
)


import matplotlib.pyplot as plt
import numpy as np

def plot_monte_carlo_results(results, param_names=None):
	"""
	Plot histograms and pairwise scatter plots of Monte Carlo fitting results.

	Args:
		results (np.ndarray): shape (n_simulations, n_parameters)
		param_names (list[str], optional): Names for each parameter. Default is generic 'Param 0', 'Param 1',...
	"""
	n_params = results.shape[1]

	if param_names is None:
		param_names = [f"Param {i}" for i in range(n_params)]

	# Histograms
	fig, axes = plt.subplots(n_params, 1, figsize=(6, 2 * n_params), tight_layout=True)
	for i in range(n_params):
		axes[i].hist(results[:, i], bins=30, color='skyblue', edgecolor='black')
		axes[i].set_title(f"Distribution of {param_names[i]}")
		axes[i].set_xlabel("Value")
		axes[i].set_ylabel("Frequency")
	plt.show()

	# Pairwise scatter plot matrix (only if less than ~6 params to keep it readable)
	if n_params <= 6:
		fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params), squeeze=False)
		for i in range(n_params):
			for j in range(n_params):
				ax = axes[i, j]
				if i == j:
					ax.hist(results[:, i], bins=30, color='lightgray')
					ax.set_xlabel(param_names[i])
				else:
					ax.scatter(results[:, j], results[:, i], s=5, alpha=0.3)
					if i == n_params - 1:
						ax.set_xlabel(param_names[j])
					if j == 0:
						ax.set_ylabel(param_names[i])
		plt.tight_layout()
		plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_fit_with_confidence_interval(df, station_name, column, fit_func, results, param_names=None, start_day=None, end_day=None, conf_level=0.95, **kwargs):
	"""
	Plot data + mean fit + confidence interval band from Monte Carlo simulation results.

	Args:
		df (pd.DataFrame): Your full data
		station_name (str): Station to plot
		column (str): Column with observed data (e.g. 'pc1')
		fit_func (callable): The model function, y = f(x, *params)
		results (np.ndarray): Monte Carlo fitted params, shape (n_simulations, n_params)
		param_names (list[str], optional): names for parameters (for plot labels)
		start_day (int, optional): start day to filter data and plot
		end_day (int, optional): end day to filter data and plot
		conf_level (float): Confidence level for the band (default 0.95)
		**kwargs: extra args passed to fit_func
	"""
	import scipy.stats

	# Filter data
	df_plot = df[df['station'] == station_name]
	if start_day is not None:
		df_plot = df_plot[df_plot['days_since_eq'] >= start_day]
	if end_day is not None:
		df_plot = df_plot[df_plot['days_since_eq'] <= end_day]

	# x, y for actual data
	x_data = df_plot['days_since_eq'].values / 365.0 / 10  # decades
	y_data = df_plot[column].values / 10.0  # cm

	# Generate smooth x grid for plotting fit + CI
	x_fit = np.linspace(min(x_data), max(x_data), 1000)

	# Predict y for each Monte Carlo parameter set
	y_preds = np.array([fit_func(x_fit, *params, **kwargs) for params in results])

	# Calculate confidence intervals
	alpha = 1 - conf_level
	lower = np.percentile(y_preds, 100 * (alpha / 2), axis=0)
	upper = np.percentile(y_preds, 100 * (1 - alpha / 2), axis=0)
	mean = np.mean(y_preds, axis=0)

	# Plot
	plt.figure(figsize=(8, 5))
	plt.scatter(x_data, y_data, color='blue', label='Data', alpha=0.7)
	plt.plot(x_fit, mean, color='red', label='Mean fit')
	plt.fill_between(x_fit, lower, upper, color='red', alpha=0.3, label=f'{int(conf_level*100)}% Confidence Interval')

	plt.title(f"Fit with confidence interval for {station_name} ({column})")
	plt.xlabel("Time since EQ (decades)")
	plt.ylabel(f"{column} (cm)")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()


plot_monte_carlo_results(results)
plot_fit_with_confidence_interval(df, station_name, 'pc1', fitting2.double_exponential_with_earthquakes_and_linear_region, results, param_names=['c1', 'm1', 'c2', 'm2', 'd', 'v'], start_day=0, end_day=365*300, earthquake_dates=eq_dates_decades)
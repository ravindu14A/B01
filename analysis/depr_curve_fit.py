import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def jacobian(x, c1, m1, c2, m2, d, v):
	"""
	Computes the Jacobian of the model with respect to the parameters.

	Args: 
	- x: input data (independent variable)
	- c1, m1, c2, m2, d, v: model parameters

	Returns:
	- A matrix of partial derivatives (Jacobian)
	"""
	# Exponential components
	exp1 = np.exp(-m1 * x)
	exp2 = np.exp(-m2 * x)

	# Partial derivatives with respect to each parameter
	dc1 = exp1
	dm1 = -c1 * x * exp1
	dc2 = exp2
	dm2 = -c2 * x * exp2
	dd = np.ones_like(x)
	dv = x

	# Jacobian matrix: each row corresponds to the partial derivatives w.r.t. each parameter
	J = np.vstack([dc1, dm1, dc2, dm2, dd]).T
	return J


def fit_linear_trend(df, station, column, start_date, end_date):
	"""
	Fits a linear model y = mx + b to the specified station's data
	between start_date and end_date using the given column.

	Returns:
		(m, b): slope and intercept
	"""
	start_date = pd.to_datetime(start_date)
	end_date = pd.to_datetime(end_date)

	# Filter data
	df_filtered = df[
		(df['station'] == station) &
		(df['date'] >= start_date) &
		(df['date'] <= end_date)
	].copy()

	if df_filtered.empty:
		raise ValueError("No data available for the specified station and date range.")

	# Convert dates to numeric (days since start_date)
	df_filtered['x'] = (df_filtered['date'] - start_date).dt.total_seconds() / (60 * 60 * 24)

	X = df_filtered[['x']].values  # 2D array
	y = df_filtered[column].values

	# Fit linear regression
	model = LinearRegression()
	model.fit(X, y)

	return model.coef_[0], model.intercept_

# Model when v is known (fixed)
def model_with_known_v(x, c1, m1, c2, m2, d, v):
	return v * x + c1 * np.exp(-m1 * x) + c2 * np.exp(-m2 * x) + d

# Model with no vx term at all
def model_without_v_term(x, c1, m1, c2, m2, d):
	return c1 * np.exp(-m1 * x) + c2 * np.exp(-m2 * x) + d

def fit_exponential_decay(df, station, start_date, end_date, column_name, known_slope=True, v=None):
	'''
	Fits an exponential decay model to time series data for a given station and date range.

	Parameters:
	-----------
	df : pandas.DataFrame
		DataFrame containing the time series data. Must include columns: 'station', 'date', and the specified column_name.

	station : str
		Name of the station to filter the data by.

	start_date : str or datetime-like
		Start date of the fitting range (inclusive).

	end_date : str or datetime-like
		End date of the fitting range (inclusive).

	column_name : str
		Name of the column in `df` containing the dependent variable to be fitted.

	known_slope : bool, optional (default=True)
		Whether to use a model with a known post-event linear slope `v`.

	v : float, optional
		Value of the known slope `v` to use if `known_slope=True`. Required in that case.

	Returns:
	--------
	popt : ndarray
		Optimal values for the parameters of the fitted model.

	pcov : 2-D ndarray
		Estimated covariance of `popt`.

	ref_date : pandas.Timestamp
		The reference date (minimum date in the filtered dataset), used as the x=0 origin for fitting.

	Raises:
	-------
	ValueError
		If the filtered DataFrame is empty or if `known_slope=True` but `v` is not provided.
	'''
	# Filter by station and date range
	mask = (
		(df['station'] == station) &
		(df['date'] >= pd.to_datetime(start_date)) &
		(df['date'] <= pd.to_datetime(end_date))
	)
	df_filtered = df.loc[mask].copy()
	if df_filtered.empty:
		raise ValueError("No data points found for the given station and date range.")

	# Compute x in days from the reference date
	df_filtered['date'] = pd.to_datetime(df_filtered['date'])
	ref_date = df_filtered['date'].min()
	df_filtered['x'] = (df_filtered['date'] - ref_date).dt.days

	x_data = df_filtered['x'].values
	y_data = df_filtered[column_name].values

	if known_slope:
		if v is None:
			raise ValueError("If known_slope=True, you must provide a value for v.")

		# Fit function with v fixed
		def fit_func(x, c1, m1, c2, m2, d):
			return model_with_known_v(x, c1, m1, c2, m2, d, v)

		initial_guess = [1.0, 0.01, 1.0, 0.001, 0.0]
		def fit_jacob(x, c1, m1, c2, m2, d):
			return jacobian(x, c1, m1, c2, m2, d,v)

	else:
		# Fit function with no vx term at all
		def fit_func(x, c1, m1, c2, m2, d):
			return model_without_v_term(x, c1, m1, c2, m2, d)

		initial_guess = [1.0, 0.01, 1.0, 0.001, 0.0]

	# Fit the model
	popt, pcov = curve_fit(fit_func, x_data, y_data, p0=initial_guess, maxfev=10000000, jac=fit_jacob)

	return popt, pcov, ref_date
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


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

	else:
		# Fit function with no vx term at all
		def fit_func(x, c1, m1, c2, m2, d):
			return model_without_v_term(x, c1, m1, c2, m2, d)

		initial_guess = [1.0, 0.01, 1.0, 0.001, 0.0]

	# Fit the model
	popt, pcov = curve_fit(fit_func, x_data, y_data, p0=initial_guess, maxfev=10000)

	return popt, pcov, ref_date

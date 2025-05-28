import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def fit_station_linear_trend(df: pd.DataFrame, station_name: str, column_name: str, start_day: int, end_day: int):
	"""
	Performs a linear fit on the specified station's data.

	Args:
		df (pd.DataFrame): Input dataframe.
		station_name (str): Name of the station to be linearly fitted.
		column_name (str): Name of the column (e.g., d_north_mm, d_east_mm, pc1, pc2).
		start_day (int): Start of range (days_since_eq).
		end_day (int): End of range (days_since_eq).

	Returns:
		Tuple[float, float]: Slope (k) and intercept (c) of the linear fit (kx + c).
	"""
	df_filtered = df[
		(df['station'] == station_name) &
		(df['days_since_eq'] >= start_day) &
		(df['days_since_eq'] <= end_day)
	]

	if len(df_filtered) <2:
		raise Exception(
	f"Not enough points to do linear interpolation for station {station_name} "
	f"on data range {start_day} - {end_day}, column '{column_name}'")			
		return 
	
	x = df_filtered['days_since_eq']
	y = df_filtered[column_name]

	k, c = np.polyfit(x, y, 1)

	return k, c


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
# Model when v is known (fixed)
def model_with_known_v(x, c1, m1, c2, m2, d, v):
	return v * x + c1 * np.exp(-m1 * x) + c2 * np.exp(-m2 * x) + d

# Model with no vx term at all
def model_without_v_term(x, c1, m1, c2, m2, d):
	return c1 * np.exp(-m1 * x) + c2 * np.exp(-m2 * x) + d

def simplified_model_with_known_v(x, c1, m1, d, v):
	return v * x + c1 * np.exp(-m1 * x) + d

def simplified_model_without_v_term(x, c1, m1, d):
	return c1 * np.exp(-m1 * x) + d


def fit_station_exponential_decay(
	df: pd.DataFrame,
	station: str,
	start_day: int,
	end_day: int,
	column_name: str,
	v: float = None,
	simplified=False,
	model_func=None,
	jac_func=None,
	initial_guess=None
):
	'''
	Fits an exponential decay model (or custom model) to time series data using `days_since_eq` for a given station and day range.

	Parameters:
	-----------
	df : pandas.DataFrame
		DataFrame containing the time series data. Must include columns: 'station', 'days_since_eq', and the specified column_name.

	station : str
		Name of the station to filter the data by.

	start_day : int
		Start of the fitting range (inclusive), in days since earthquake.

	end_day : int
		End of the fitting range (inclusive), in days since earthquake.

	column_name : str
		Name of the column in `df` containing the dependent variable to be fitted.

	v : float, optional
		If provided, uses a model with known linear post-event slope `v`. If None, uses a model without a slope term.

	simplified : bool, optional
		If True, uses a simplified model with fewer parameters.

	model_func : callable, optional
		Custom model function to use for fitting. If provided, overrides built-in models.

	jac_func : callable, optional
		Custom Jacobian function to use for fitting. If provided, overrides built-in Jacobians.

	initial_guess : list or ndarray, optional
		Initial guess for the parameters. If provided, overrides built-in guesses.

	Returns:
	--------
	popt : ndarray
		Optimal values for the parameters of the fitted model.

	pcov : 2-D ndarray
		Estimated covariance of `popt`.

	Raises:
	-------
	ValueError
		If the filtered DataFrame is empty.
	'''
	# Filter by station and day range
	df_filtered = df[
		(df['station'] == station) &
		(df['days_since_eq'] >= start_day) &
		(df['days_since_eq'] <= end_day)
	].copy()

	if df_filtered.empty:
		raise ValueError("No data points found for the given station and day range.")

	x_data = df_filtered['days_since_eq'].values
	y_data = df_filtered[column_name].values

	# Use custom model if provided
	if model_func is not None:
		fit_func = model_func
		fit_jacob = jac_func
		guess = initial_guess
	else:
		# Decide which built-in model to use
		if not simplified:
			if v is not None:
				def fit_func(x, c1, m1, c2, m2, d):
					return model_with_known_v(x, c1, m1, c2, m2, d, v)

				def fit_jacob(x, c1, m1, c2, m2, d):
					return jacobian(x, c1, m1, c2, m2, d, v)
			else:
				def fit_func(x, c1, m1, c2, m2, d):
					return model_without_v_term(x, c1, m1, c2, m2, d)

				fit_jacob = None  # No Jacobian if not using v
			guess = [4.46887393e+02, 2.28338188e-04, 1.13055458e+02, 4.34198176e-03, -7.15547149e+02]
		else:
			if v is not None:
				def fit_func(x, c1, m1, d):
					return simplified_model_with_known_v(x, c1, m1, d, v)

				def fit_jacob(x, c1, m1, d):
					return jacobian(x, c1, m1, 1, 1, d, v)  # Placeholder
			else:
				def fit_func(x, c1, m1, d):
					return simplified_model_without_v_term(x, c1, m1, d)

				fit_jacob = None  # No Jacobian if not using v
			guess = [1, 0.01, -200]

	if initial_guess is not None:
		guess = initial_guess

	try:
		popt, pcov = curve_fit(
			fit_func,
			x_data,
			y_data,
			p0=guess,
			maxfev=10000,
			jac=fit_jacob
		)
		#print("in fit popt", popt)
	except Exception as e:
		print("Curve fitting failed:", e)
		return None, None
	return popt, pcov

from analysis.fitting import model_without_v_term, model_with_known_v
import pandas as pd
from analysis import fitting

def fit_model_and_find_zero(df: pd.DataFrame, station_name: str, column_name: str, v: float = None):
	"""
	Fits an exponential decay model to station data and calculates when the model reaches zero.
	
	Parameters:
	-----------
	df : pandas.DataFrame
		DataFrame containing the station data
	station_name : str
		Name of the station to fit
	column_name : str
		Column name to fit (e.g., 'pc1', 'east', 'north')
	v : float, optional
		Known velocity parameter for model fitting
		
	Returns:
	--------
	tuple:
		popt : ndarray
			Optimal values for the model parameters (c1, m1, c2, m2, d)
		pcov : 2-D ndarray
			Estimated covariance of the parameters
		days_to_reach_0 : float or None
			Number of days after the earthquake when the model reaches zero,
			or None if the model never reaches zero or if fitting failed
	"""
	# Find the maximum end date for this station
	station_mask = df['station'] == station_name
	max_end_date = df[station_mask]['days_since_eq'].max()
	
	# Fit the exponential decay model
	try:
		popt, pcov = fitting.fit_station_exponential_decay(
			df, 
			station_name, 
			start_day=0, 
			end_day=max_end_date, 
			column_name=column_name, 
			v=v
		)
	except Exception as e:
		print(f"Error fitting model for station {station_name}: {e}")
		return None, None, None
	
	# Check if fitting was successful
	if popt is None:
		print(f"Model fitting failed for station {station_name}")
		return None, None, None
	
	# Define a function to find where the model equals zero
	def model_minus_zero(x):
		if v is not None:
			return model_with_known_v(x, *popt, v)
		else:
			return model_without_v_term(x, *popt)
	from scipy.optimize import brentq

	# Your model function: model_with_known_v(x, *popt, v)
	# Assume it's already defined

	# Wrapper for root finding — fixes parameters popt and v

	# Define bounds
	x_lower = 100
	x_upper = 365 * 100000

	# Find the root
	x_root = None
	try:
		x_root = brentq(model_minus_zero, x_lower, x_upper)
		#print(f"Root found at x = {x_root}")
	except ValueError as e:
		print("No root found in the interval:", e)
	
	return popt, pcov, x_root
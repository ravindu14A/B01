import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from functools import partial
import matplotlib.pyplot as plt

def single_exponential_with_earthquakes(x, c, m, d, *eq_params, earthquake_dates):
    # eq_params is a tuple of parameters (length = number of earthquakes)
    # earthquake_dates is an array of the same length
    y = c * np.exp(-m * x) + d
    for eq_param, eq_date in zip(eq_params, earthquake_dates):
        y += eq_param * np.heaviside(x - eq_date, 0)
    return y


def jacobian_single_exponential_with_earthquakes(x, c, m, d, *eq_params, earthquake_dates):
    """
    Jacobian of:
        f(x) = c * exp(-m * x) + d + sum(eq_param_i * heaviside(x - earthquake_date_i))

    Parameters:
        x               : array-like
        c, m, d         : model parameters
        eq_params       : tuple of eq_param for each earthquake (variable length)
        earthquake_dates: array of earthquake dates (fixed, not optimized)

    Returns:
        J : Jacobian matrix of shape (len(x), 3 + len(eq_params)),
            columns correspond to partial derivatives w.r.t [c, m, d, eq_param_1, eq_param_2, ...]
    """
    x = np.asarray(x)
    n = len(x)
    n_eq = len(eq_params)

    J = np.zeros((n, 3 + n_eq))

    exp_term = np.exp(-m * x)

    # Partial derivatives of the core exponential model
    J[:, 0] = exp_term                # ∂f/∂c
    J[:, 1] = -c * x * exp_term      # ∂f/∂m
    J[:, 2] = 1                      # ∂f/∂d

    # Partial derivatives w.r.t. eq_params
    for i, eq_date in enumerate(earthquake_dates):
        heav = np.heaviside(x - eq_date, 0)
        J[:, 3 + i] = heav           # ∂f/∂eq_param_i

    return J

def double_exponential_no_v_term_and_earthquakes(x, c1, m1, c2, m2, d, eq_param, earthquake_date):
	'''
	Double exponential model with earthquake correction:
	c1 * exp(-m1 * x) + c2 * exp(-m2 * x) + d + eq_param * heaviside(x - earthquake_date)
	'''
	return (
		c1 * np.exp(-m1 * x)
		+ c2 * np.exp(-m2 * x)
		+ d
		+ eq_param * np.heaviside(x - earthquake_date, 0)
	)

def jacobian_double_exponential_no_v_term_and_earthquakes(x, c1, m1, c2, m2, d, eq_param, earthquake_date):
	"""
	Jacobian of: f(x) = c1 * exp(-m1 * x) + c2 * exp(-m2 * x) + d + eq_param * heaviside(x - earthquake_date)

	Parameters:
		x : array-like
		c1, m1, c2, m2, d, eq_param : model parameters
		earthquake_date : fixed value (not optimized)

	Returns:
		Jacobian matrix of shape (len(x), 6), where columns are partials w.r.t [c1, m1, c2, m2, d, eq_param]
	"""
	x = np.asarray(x)
	J = np.zeros((len(x), 6))

	exp1 = np.exp(-m1 * x)
	exp2 = np.exp(-m2 * x)

	J[:, 0] = exp1                        # ∂f/∂c1
	J[:, 1] = -c1 * x * exp1              # ∂f/∂m1
	J[:, 2] = exp2                        # ∂f/∂c2
	J[:, 3] = -c2 * x * exp2              # ∂f/∂m2
	J[:, 4] = 1                           # ∂f/∂d
	J[:, 5] = np.heaviside(x - earthquake_date, 0)  # ∂f/∂eq_param
	return J


def single_exponential(x, c, m, d):
	"""
	Single exponential model: c * exp(-m * x) + d
	"""
	return c * np.exp(-m * x) + d

def jacobian_single_exponential(x, c, m, d):
	"""
	Jacobian matrix of single_exponential with respect to parameters [c, m, d].
	Returns an array of shape (len(x), 3).
	"""
	jac = np.zeros((x.size, 3))
	exp = np.exp(-m * x)
	jac[:, 0] = exp                      # d/dc
	jac[:, 1] = -c * x * exp              # d/dm
	jac[:, 2] = 1                         # d/dd
	return jac

def double_exponential(x, c1, m1, c2, m2, d):
	"""
	c1 * exp(-m1 * x) + c2 * exp(-m2 * x) + d
	Parameters:
		c1, m1: Coefficient and decay rate for the first exponential term.
		c2, m2: Coefficient and decay rate for the second exponential term.
		d: Constant offset.
	Returns:	
		an array of shape (len(x),).
	"""
	return c1 * np.exp(-m1 * x) + c2 * np.exp(-m2 * x) + d

def jacobian_double_exponential(x, c1, m1, c2, m2, d):
	"""
	Jacobian matrix of double_exponential with respect to parameters [c1, m1, c2, m2, d].
	Returns an array of shape (len(x), 5).
	"""
	jac = np.zeros((x.size, 5))
	exp1 = np.exp(-m1 * x)
	exp2 = np.exp(-m2 * x)
	jac[:, 0] = exp1                      # d/dc1
	jac[:, 1] = -c1 * x * exp1            # d/dm1
	jac[:, 2] = exp2                      # d/dc2
	jac[:, 3] = -c2 * x * exp2            # d/dm2
	jac[:, 4] = 1                         # d/dd
	return jac


def single_exponential_v(x, c, m, d, v):
	return c * np.exp(-m * x) + d + v * x

def jacobian_single_exponential_v(x, c, m, d, v):
	"""
	Jacobian matrix of single_exponential_with_v_term with respect to parameters [c, m, d].
	Returns an array of shape (len(x), 3).
	"""
	jac = np.zeros((x.size, 3))
	exp = np.exp(-m * x)
	jac[:, 0] = exp                      # d/dc
	jac[:, 1] = -c * x * exp              # d/dm
	jac[:, 2] = 1                         # d/dd
	return jac

def double_exponential_v(x, c1, m1, c2, m2, d, v):
	return c1 * np.exp(-m1 * x) + c2 * np.exp(-m2 * x) + d + v * x

def jacobian_double_exponential_v(x, c1, m1, c2, m2, d, v):
	"""
	Jacobian matrix of model_with_v_term with respect to parameters [c1, m1, c2, m2, d].
	Returns an array of shape (len(x), 5).
	"""
	jac = np.zeros((x.size, 5))
	exp1 = np.exp(-m1 * x)
	exp2 = np.exp(-m2 * x)
	jac[:, 0] = exp1                      # d/dc1
	jac[:, 1] = -c1 * x * exp1            # d/dm1
	jac[:, 2] = exp2                      # d/dc2
	jac[:, 3] = -c2 * x * exp2            # d/dm2
	jac[:, 4] = 1                         # d/dd
	return jac

def fit_station_with_model(df : pd.DataFrame, station_name : str, column : str,
						   start_day :int, end_day:int,
						   fit_func, fit_jacob : int = None, initial_guess: list = None, bounds=None,
						   **kwargs):
	"""
	Fits the given function to the data for a specific station between start and end indices,
	and plots the fit.

	Parameters:
		df (pd.DataFrame): DataFrame containing the data.
		station_name (str): Name of the station (column in df).
		start_day (int): Start time (in days since EQ) for fitting.
		end_day (int): End time (in days since EQ) for fitting.
		fit_func (callable): Function to fit (e.g., linear, exponential).
		**kwargs: Parameters passed to fit_func.

	Returns:
		popt (array): Optimal values for the parameters.
		pcov (2d array): Covariance of popt.
	"""
	df_filtered = df[
		(df['station'] == station_name) &
		(df['days_since_eq'] >= start_day) &
		(df['days_since_eq'] <= end_day)
	]
	
	# Convert to decades and cm
	x = df_filtered['days_since_eq'].values / 365.0 / 10
	y = df_filtered[column].values / 10.0  # mm to meters

	print("DEBUG, length of array is", len(x))

	wrapped_func = partial(fit_func, **kwargs)
	wrapped_jacob = partial(fit_jacob, **kwargs)
	if bounds is not None:
		popt, pcov, full_output, mesg, ier = curve_fit(wrapped_func, x, y, jac=wrapped_jacob, p0=initial_guess, maxfev=100000, full_output = True, bounds = bounds)
	else:
		popt, pcov, full_output, mesg, ier = curve_fit(wrapped_func, x, y, jac=wrapped_jacob, p0=initial_guess, maxfev=100000, full_output = True)

	### --- Plotting the Fit ---
	plt.figure(figsize=(8, 5))
	plt.scatter(x, y, color='blue', label='Data', alpha=0.7)

	# Generate smooth x for the fit line
	x_fit = np.linspace(min(x), max(x), 500)
	y_fit = wrapped_func(x_fit, *popt)
	plt.plot(x_fit, y_fit, color='red', label='Fitted model')

	plt.title(f"Fit for {station_name} ({column})")
	plt.xlabel("Time since EQ (decades)")
	plt.ylabel(f"{column} (cm)")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()
	### ----------------------

	return popt, pcov, full_output, mesg, ier

from scipy.optimize import curve_fit
import numpy as np
from functools import partial
import pandas as pd

def prepare_monte_carlo_inputs(df, station_name, column, start_day, end_day):
	df_filtered = df[
		(df['station'] == station_name) &
		(df['days_since_eq'] >= start_day) &
		(df['days_since_eq'] <= end_day)
	].copy()

	x = df_filtered['days_since_eq'].values / 365.0 / 10

	if column != "pc1":
		raise ValueError("Only 'pc1' is supported for now.")

	y = df_filtered[column].values / 10.0  # mm to cm
	sigma = df_filtered['pca_covariance_pc_space'].apply(lambda cov: np.sqrt(cov[0, 0])).values / 10.0

	return x, y, sigma


def simulate_single_fit(x, y, sigma, fit_func, initial_guess=None, bounds=None, fit_jacob=None, **kwargs):
	wrapped_func = partial(fit_func, **kwargs)
	wrapped_jacob = partial(fit_jacob, **kwargs) if fit_jacob else None

	# Add Gaussian noise
	perturbed_y = y + np.random.normal(0, sigma)

	try:
		if bounds is not None:
			popt, _ = curve_fit(
				wrapped_func, x, perturbed_y, p0=initial_guess,
				bounds=bounds, jac=wrapped_jacob,
				sigma=sigma, absolute_sigma=True, maxfev=100000
			)
		else:
			popt, _ = curve_fit(
				wrapped_func, x, perturbed_y, p0=initial_guess,
				jac=wrapped_jacob,
				sigma=sigma, absolute_sigma=True, maxfev=100000
			)
	except Exception as e:
		print("Fit failed:", e)
		return None

	return popt

from joblib import Parallel, delayed

def run_monte_carlo_fit(x, y, sigma, fit_func, initial_guess, bounds=None, fit_jacob=None,
						n_simulations=1000, n_jobs=-1, **kwargs):
	results = Parallel(n_jobs=n_jobs)(
		delayed(simulate_single_fit)(
			x, y, sigma, fit_func, initial_guess, bounds, fit_jacob, **kwargs
		) for _ in range(n_simulations)
	)

	# Drop failed fits
	results = [r for r in results if r is not None]
	return np.array(results)

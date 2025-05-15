import pandas as pd
import numpy as np
from analysis import fitting
from analysis.fitting import fit_station_linear_trend
import matplotlib.pyplot as plt

def depr_detect_and_remove_eq(df: pd.DataFrame, station_name: str, column: str,
						 filtered_eq_days_since: int, v=None,
						 n_detection_points: int = 7,
						 visualize: bool = True,
						 avg_residual_multiplier: float = 1.5):
	"""
	Detect and potentially remove earthquake signals in the data using average residuals.
	"""
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt

	# Filter for the relevant station
	df_station = df[df['station'] == station_name].copy()

	# Use only data before filtered_eq_days_since for fitting
	df_before = df_station[
		(df_station['days_since_eq'] < filtered_eq_days_since) & 
		(df_station['days_since_eq'] > 0)
	].copy()

	if len(df_before) < 8:
		return False, df

	# Add zero point
	zero_point = pd.DataFrame({'station': [station_name], 'days_since_eq': [0], column: [0]})
	for col in zero_point.columns:
		if col not in df_before.columns:
			df_before[col] = station_name if col == 'station' else None
	df_before = pd.concat([zero_point, df_before], ignore_index=True)

	if len(df_before) < 8:
		return False, df

	# Fit decay model
	popt, _ = fitting.fit_station_exponential_decay(
		df_before, station_name, 0, filtered_eq_days_since, column, v=v)

	x_fit = df_before['days_since_eq'].values
	if v is not None:
		y_fit = np.array([fitting.model_with_known_v(x, *popt, v=v) for x in x_fit])
	else:
		y_fit = np.array([fitting.model_without_v_term(x, *popt) for x in x_fit])

	residuals_fit = df_before[column].values - y_fit
	std_residuals = np.std(residuals_fit)

	# Get next N points after fitting window
	df_after = df_station[df_station['days_since_eq'] >= filtered_eq_days_since].copy()
	df_after = df_after.sort_values('days_since_eq').head(n_detection_points)

	if len(df_after) < n_detection_points:
		return False, df

	# Predict expected values and compute residuals
	x_detect = df_after['days_since_eq'].values
	if v is not None:
		expected = np.array([fitting.model_with_known_v(x, *popt, v=v) for x in x_detect])
	else:
		expected = np.array([fitting.model_without_v_term(x, *popt) for x in x_detect])

	df_after['expected'] = expected
	df_after['residual'] = df_after[column] - df_after['expected']
	mean_abs_residual = np.mean(np.abs(df_after['residual']))
	threshold = avg_residual_multiplier * std_residuals

	# New detection logic: smoothed average above threshold
	earthquake_detected = mean_abs_residual > threshold

	# Visualization
	if visualize:
		plt.figure(figsize=(10, 6))
		plt.scatter(df_before['days_since_eq'], df_before[column], label='Fitting Data', color='blue')
		plt.plot(x_fit, y_fit, label='Fitted Model', color='black')
		plt.scatter(df_after['days_since_eq'], df_after[column], label='Detection Points', color='orange')
		plt.plot(df_after['days_since_eq'], df_after['expected'], '--', color='red', label='Expected (Detection)')
		plt.axvline(x=filtered_eq_days_since, linestyle=':', color='gray', label='Detection Start')
		plt.title(f"Detection at {station_name}")
		plt.xlabel("Days Since Earthquake")
		plt.ylabel(column)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.show()

		# Residual plot
		plt.figure(figsize=(10, 3))
		plt.plot(df_after['days_since_eq'], df_after['residual'], label='Detection Residuals', color='purple', marker='o')
		plt.axhline(threshold, linestyle='--', color='red', label=f'±{avg_residual_multiplier}×STD')
		plt.axhline(-threshold, linestyle='--', color='red')
		plt.xlabel("Days Since Earthquake")
		plt.ylabel("Residual")
		plt.title("Detection Residuals")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.show()

	return earthquake_detected, df




def correct_earthquake_signal_curve_fitting(
	df: pd.DataFrame,
	station_name: str,
	column: str,
	filtered_eq_days_since: int,
	v=None,
	shift_fit_offset: int = 180,
	shift_fit_duration: int = 90,
	visualize: bool = False
) -> pd.DataFrame:
	"""
	Corrects earthquake signal in the whole DataFrame by:
	- fitting model to pre-earthquake data of given station,
	- dropping first 6 months post-earthquake data for that station,
	- vertically shifting the rest of that station's post-6-month data to minimize error,
	- optionally visualizing the fit and corrected data,
	- returning the whole DataFrame with corrected values for the station only.
	"""
	# Extract station-specific data
	df_station = df[df['station'] == station_name].copy()

	# Fit model on data before the earthquake (excluding 6 months after)
	df_before = df_station[
		(df_station['days_since_eq'] < filtered_eq_days_since) & 
		(df_station['days_since_eq'] > 0)
	].copy()

	if len(df_before) < 8:
		return df

	# Fit decay model
	popt, _ = fitting.fit_station_exponential_decay(
		df_before, station_name, 0, filtered_eq_days_since, column, v=v
	)
	#check if unable to do decay
	if popt is None:
		print("unable to do fitting of func, skipping for station", station_name, "and earthquake", filtered_eq_days_since)
		return df

	# Evaluate shift fit using 6–9 month window
	start_eval = filtered_eq_days_since + shift_fit_offset
	end_eval = start_eval + shift_fit_duration
	df_eval = df_station[
		(df_station['days_since_eq'] >= start_eval) &
		(df_station['days_since_eq'] < end_eval)
	].copy()

	if df_eval.empty:
		return df

	# Calculate expected values for pre-EQ data (for visualization)
	x_before = df_before['days_since_eq'].values
	expected_before = np.array([
		fitting.model_with_known_v(x, *popt, v=v) if v is not None
		else fitting.model_without_v_term(x, *popt)
		for x in x_before
	])

	# Calculate expected values for evaluation window
	x_eval = df_eval['days_since_eq'].values
	expected_eval = np.array([
		fitting.model_with_known_v(x, *popt, v=v) if v is not None
		else fitting.model_without_v_term(x, *popt)
		for x in x_eval
	])
	observed_eval = df_eval[column].values

	# Optimize vertical shift
	from scipy.optimize import minimize_scalar
	def mse_shift(d):
		return np.mean((observed_eval - (expected_eval + d))**2)

	res = minimize_scalar(mse_shift)
	best_shift = res.x

	# Drop 6 months post-EQ data for this station
	to_drop_mask = (
		(df['station'] == station_name) &
		(df['days_since_eq'] > filtered_eq_days_since) &
		(df['days_since_eq'] <= filtered_eq_days_since + shift_fit_offset)
	)
	df = df.loc[~to_drop_mask].copy()

	# Apply vertical shift to post-6-month data for this station
	post_shift_mask = (df['station'] == station_name) & (df['days_since_eq'] > filtered_eq_days_since + shift_fit_offset)
	df.loc[post_shift_mask, column] -= best_shift

	# Visualization
	if visualize:
		plt.figure(figsize=(10, 6))

		# Fitted decay model: smooth line from 0 to max days_since_eq of station
		x_smooth = np.linspace(0, df_station['days_since_eq'].max(), 300)
		y_smooth = np.array([
			fitting.model_with_known_v(x, *popt, v=v) if v is not None
			else fitting.model_without_v_term(x, *popt)
			for x in x_smooth
		])
		plt.plot(x_smooth, y_smooth, label='Fitted Decay Model', color='blue')

		# Pre-earthquake points
		plt.scatter(x_before, df_before[column], label='Pre-EQ Data', color='black')
		# Dropped points 0-6 months after EQ
		df_dropped = df_station[
			(df_station['days_since_eq'] > filtered_eq_days_since) &
			(df_station['days_since_eq'] <= filtered_eq_days_since + shift_fit_offset)
		]
		plt.scatter(df_dropped['days_since_eq'], df_dropped[column], label='Dropped 0-6mo Data', color='orange')

		# Evaluation window observed points before shift
		plt.scatter(x_eval, observed_eval, label='6-9mo Eval Observed (Before Shift)', color='red', alpha=0.7)

		# Evaluation window observed points after shift
		plt.scatter(x_eval, observed_eval - best_shift, label='6-9mo Eval Observed (After Shift)', color='green', alpha=0.7)
		"""		
		print("Pre-EQ Data:")
		for day, val in zip(x_before, df_before[column]):
			print(f"{val}, {day}")

		print("\nDropped 0-6mo Data:")
		for day, val in zip(df_dropped['days_since_eq'], df_dropped[column]):
			print(f"{val}, {day}")
		
		print("\n6-9mo Eval Observed (After Shift):")
		for day, val in zip(x_eval, observed_eval - best_shift):
			print(f"{val}, {day}")

		"""		
		plt.axvline(filtered_eq_days_since, color='black', linestyle=':', label='Earthquake Date')
		plt.xlabel('Days Since Earthquake')
		plt.ylabel(column)
		plt.title(f'Station: {station_name} Earthquake Correction')
		plt.legend()
		plt.grid(True)
		plt.show()

	return df.sort_values(['station', 'days_since_eq']).reset_index(drop=True)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def correct_earthquake_signal_simple_shift(
	df: pd.DataFrame,
	station_name: str,
	column: str,
	filtered_eq_days_since: int,
	pre_window: int = 30,
	post_window: int = 15,
	delete_post_window: int = 30,
	visualize: bool = False
) -> pd.DataFrame:

	df_corrected = df.copy()
	df_station = df_corrected[df_corrected['station'] == station_name]

	# Select pre-EQ data
	pre_mask = (df_station['days_since_eq'] >= filtered_eq_days_since - pre_window) & \
	           (df_station['days_since_eq'] < filtered_eq_days_since)
	pre_df = df_station.loc[pre_mask, ['days_since_eq', column]]

	if len(pre_df) < 2:
		print(f"[DEBUG] Not enough pre-EQ points for station {station_name}")
		return df_corrected

	# Fit a line to pre-EQ data
	x_pre = pre_df['days_since_eq'].values
	y_pre = pre_df[column].values
	slope, intercept = np.polyfit(x_pre, y_pre, 1)
	expected_post_eq_value = slope * filtered_eq_days_since + intercept

	# Actual post-EQ average (close window to avoid long-term decay)
	post_mask = (df_station['days_since_eq'] >= filtered_eq_days_since) & \
	            (df_station['days_since_eq'] < filtered_eq_days_since + post_window)
	post_vals = df_station.loc[post_mask, column]

	if post_vals.empty:
		print(f"[DEBUG] No post-EQ data to correct for station {station_name}")
		return df_corrected

	post_avg = post_vals.mean()
	shift = post_avg - expected_post_eq_value

	print(f"[DEBUG] Fitted Pre-EQ trend predicts: {expected_post_eq_value:.4f}, Actual Post-EQ avg: {post_avg:.4f}, Shift: {shift:.4f}")

	# Apply shift
	mask_shift = (df_corrected['station'] == station_name) & \
	             (df_corrected['days_since_eq'] >= filtered_eq_days_since)
	df_corrected.loc[mask_shift, column] -= shift

	# Delete points in early post-EQ window
	delete_mask = (df_corrected['station'] == station_name) & \
	              (df_corrected['days_since_eq'] >= filtered_eq_days_since) & \
	              (df_corrected['days_since_eq'] < filtered_eq_days_since + delete_post_window)
	df_corrected = df_corrected.loc[~delete_mask].reset_index(drop=True)

	# Visualization
	if visualize:
		plt.figure(figsize=(10, 5))
		plt.scatter(df[df['station'] == station_name]['days_since_eq'], df[df['station'] == station_name][column], label='Original Data', alpha=0.5)
		plt.scatter(df_corrected[df_corrected['station'] == station_name]['days_since_eq'], df_corrected[df_corrected['station'] == station_name][column], 
		            label='Corrected Data', color='orange', alpha=0.7)
		plt.axvline(filtered_eq_days_since, color='red', linestyle='--', label='EQ Day')
		plt.plot(x_pre, slope * x_pre + intercept, color='green', linestyle='--', label='Pre-EQ Trend')
		plt.axhline(expected_post_eq_value, color='green', linestyle=':', label='Expected Post-EQ Level')
		plt.axvspan(filtered_eq_days_since, filtered_eq_days_since + delete_post_window, color='red', alpha=0.1, label='Deleted Interval')
		plt.xlabel('Days Since EQ')
		plt.ylabel(column)
		plt.title(f'{station_name} - Shift using Pre-EQ Trend')
		plt.legend()
		plt.show()

	return df_corrected

def test_eq_fit(df: pd.DataFrame, station_name: str, column: str, filtered_eq_days_since: int, v=None, save_points=False):
	"""
	Test and visualize the exponential decay fitting for a station.
	
	Parameters:
	-----------
	df : pd.DataFrame
		DataFrame containing the data
	station_name : str
		Name of the station to analyze
	column : str
		Column name containing the measurement values
	filtered_eq_days_since : int
		Number of days since earthquake to use for fitting the decay model
	v : float, optional
		Fixed parameter for the model with known v
	save_points : bool, optional
		If True, save data points to a text file for external plotting
	
	Returns:
	--------
	tuple
		(popt, pcov) - The fitted parameters and covariance matrix
	"""
	# Filter data for the station
	df_station = df[df['station'] == station_name].copy()
	df_before = df_station[
		(df_station['days_since_eq'] < filtered_eq_days_since) & 
		(df_station['days_since_eq'] > 0)
	].copy()
	
	# Add artificial point (0,0) at earthquake date
	zero_point = pd.DataFrame({'station': [station_name], 'days_since_eq': [0], column: [0]})
	df_before = pd.concat([zero_point, df_before], ignore_index=True)
	
	# Fit exponential decay model to the data
	popt, pcov = fitting.fit_station_exponential_decay(
		df_before, station_name, 0, filtered_eq_days_since, column, v=v)
	
	# Print the fitted parameters
	print(f"Fitted parameters: {popt}")
	print(f"\nFor Desmos equations:")
	if v is not None:
		c1, m1, c2, m2, d = popt
		print(f"{v}x + {c1}e^(-{m1}x) + {c2}e^(-{m2}x) + {d}")
	else:
		c1, m1, c2, m2, d = popt
		print(f"{c1}e^(-{m1}x) + {c2}e^(-{m2}x) + {d}")
	
	# Print the data points in a format suitable for copy-pasting into Desmos
	print("\nData points for Desmos (x,y format):")
	desmos_points = ""
	for x, y in zip(df_before['days_since_eq'], df_before[column]):
		desmos_points += f"({x},{y})\n"
	print(desmos_points)
	
	# Save points to file if requested
	if save_points:
		with open(f"{station_name}_{column}_points.txt", "w") as f:
			# Header
			f.write(f"days_since_eq,{column}\n")
			# Points
			for x, y in zip(df_before['days_since_eq'], df_before[column]):
				f.write(f"{x},{y}\n")
		print(f"Points saved to {station_name}_{column}_points.txt")
	
	# Create visualization
	import matplotlib.pyplot as plt
	
	# Create a smooth curve for the model
	x_smooth = np.linspace(0, filtered_eq_days_since * 1.5, 1000)
	
	if v is not None:
		# Model with known v parameter
		y_smooth = np.array([fitting.model_with_known_v(x, *popt, v=v) for x in x_smooth])
	else:
		# Model without v term
		y_smooth = np.array([fitting.model_without_v_term(x, *popt) for x in x_smooth])
	
	# Plot the data and the fitted curve
	plt.figure(figsize=(12, 8))
	plt.scatter(df_before['days_since_eq'], df_before[column], color='blue', label='Data points')
	plt.plot(x_smooth, y_smooth, 'r-', label='Fitted model')
	
	# Add labels and title
	plt.xlabel('Days since earthquake')
	plt.ylabel(column)
	plt.title(f'Station {station_name}: Exponential Decay Fit')
	plt.grid(True, alpha=0.3)
	plt.legend()
	
	# Show the plot
	plt.show()
	
	return popt, pcov

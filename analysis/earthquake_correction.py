import pandas as pd
import numpy as np
from scipy.optimize import minimize
from analysis.curve_fit import fit_exponential_decay, model_without_v_term

def remove_earthquake_column(df, station, date, column):
	quake_date = pd.to_datetime(date)
	df['date'] = pd.to_datetime(df['date'])  # ensure datetime

	# Important protected date
	protected_date = pd.Timestamp("2004-12-26")

	# Define windows
	pre_start = quake_date - pd.Timedelta(days=90)
	post_start = quake_date + pd.Timedelta(days=180)
	post_end = post_start + pd.Timedelta(days=90)

	# Check if protected date is in the full modeled range
	if pre_start <= protected_date <= post_end:
		raise ValueError("Data range includes Dec 26, 2004 — protected earthquake event.")

	# --- 1. Get pre-earthquake data
	df_pre = df[
		(df['station'] == station) &
		(df['date'] >= pre_start) &
		(df['date'] < quake_date)
	]
	if len(df_pre) < 10:
		raise ValueError("Not enough pre-earthquake points (need at least 10 in last 3 months).")

	# --- 2. Get post-earthquake data
	df_post = df[
		(df['station'] == station) &
		(df['date'] >= post_start) &
		(df['date'] <= post_end)
	]
	if len(df_post) < 10:
		raise ValueError("Not enough post-earthquake points (need at least 10 between 6–9 months after quake).")

	# --- 3. Fit model to pre-quake data
	popt, _, ref_date = fit_exponential_decay(
		df_pre,
		station,
		start_date=pre_start,
		end_date=quake_date,
		column_name=column,
		known_slope=False
	)

	# --- 4. Predict and compute offset
	x_post_days = (df_post['date'] - ref_date).dt.days
	y_post = df_post[column].values
	y_model = model_without_v_term(x_post_days, *popt)

	def error(offset):
		return np.mean((y_post - (y_model + offset)) ** 2)

	offset = minimize(error, x0=[0]).x[0]

	# --- 5. Apply correction
	mask_after_quake = (df['station'] == station) & (df['date'] > quake_date)
	df.loc[mask_after_quake, column] -= offset

	# --- 6. Remove data from 6-month window immediately after quake
	remove_end = quake_date + pd.Timedelta(days=180)
	df = df[~((df['station'] == station) & (df['date'] > quake_date) & (df['date'] <= remove_end))]

	return df

import pandas as pd
from analysis.curve_fit import fit_linear_trend, fit_exponential_decay

def find_curve_for_station(df, station, column):
	"""
	Fits a decay model to station data up to the 2004-12-26 earthquake,
	using the linear slope from an earlier period as known velocity.
	Returns: popt, reference_date
	"""
	# 1. Filter station data
	df_station = df[df['station'] == station].copy()
	df_station['date'] = pd.to_datetime(df_station['date'])

	if df_station.empty:
		raise ValueError("No data available for the given station.")

	# 2. Identify time window
	start_date = df_station['date'].min()
	quake_date = pd.Timestamp("2004-12-26")
	cutoff_date = quake_date #there is not data for any station on that date so its okay to do it like this 

	# 3. Estimate velocity via linear regression
	v, _ = fit_linear_trend(df, station, column, start_date, cutoff_date)


	
	# 4. Fit full decay model up to and including the earthquake
	popt, pcov, ref_date = fit_exponential_decay(
		df_station,
		station,
		start_date=quake_date,
		end_date=df_station['date'].max(),
		column_name=column,
		known_slope=True,
		v=v
	)

	return popt, pcov, ref_date, v 

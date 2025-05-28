import pandas as pd
#from analysis.depr_curve_fit import fit_linear_trend
from datetime import datetime
from analysis.fitting import fit_station_linear_trend

def get_reference_value(df, station, column, eq_day=0, min_required=5):
	"""
	Computes a reference displacement value for a station's component (e.g., d_north_mm).

	- If the station has at least `min_required` data points before the earthquake, 
	  fits a linear trend and evaluates it at eq_day.
	- Otherwise, falls back to the earliest available displacement value.

	Args:
		df (pd.DataFrame): DataFrame containing station data.
		station (str): Station name.
		column (str): Displacement column name to process.
		eq_day (int): Earthquake day in 'days_since_eq' (default: 0).
		min_required (int): Minimum number of data points required for linear fitting.

	Returns:
		float: Reference displacement value at eq_day.
	"""
	station_df = df[df['station'] == station]
	pre_eq = station_df[station_df['days_since_eq'] < eq_day]

	if len(pre_eq) < min_required:
		return station_df[column].iloc[0]

	min_day = pre_eq['days_since_eq'].min()

	k, c = fit_station_linear_trend(
		df,
		station_name = station,
		column_name = column,
		start_day = min_day,
		end_day = eq_day - 1
	)

	return k * eq_day + c


def center_station_displacements_with_trend(df, station, eq_day=0, min_required=5):
	"""
	Recenters the displacement values ('d_north_mm', 'd_east_mm') for a single station.

	Uses a linear fit before the earthquake if sufficient data exists, 
	otherwise falls back to earliest displacement value.

	Args:
		df (pd.DataFrame): DataFrame containing data for a single station.
		station (str): Station name.
		eq_day (int): Earthquake day in 'days_since_eq' (default: 0).
		min_required (int): Minimum number of points for linear fit (default: 5).

	Returns:
		pd.DataFrame: A new DataFrame with displacement columns centered.
	"""
	df = df.copy()

	for col in ['d_north_mm', 'd_east_mm']:
		eq_val = get_reference_value(df, station, col, eq_day, min_required)
		df[col] = df[col] - eq_val

	return df


def center_all_stations_with_trend(df, station_col='station', eq_day=0, min_required=5):
	"""
	Applies displacement centering to all stations in the DataFrame.

	For each station:
	- If there are at least `min_required` pre-earthquake points, a linear trend is used to set the earthquake date as 0,0 point in north and east.
	- Otherwise, the earliest value is used as the reference.

	Args:
		df (pd.DataFrame): DataFrame with station displacement data.
		station_col (str): Column name for station identifiers.
		eq_day (int): Earthquake day in 'days_since_eq'.
		min_required (int): Minimum number of points for linear fit.

	Returns:
		pd.DataFrame: New DataFrame with displacement columns centered.
	"""
	centered_dfs = []

	for station_name in df[station_col].unique():
		station_df = df[df[station_col] == station_name].copy()
		centered_df = center_station_displacements_with_trend(
			station_df,
			station=station_name,
			eq_day=eq_day,
			min_required=min_required
		)
		centered_dfs.append(centered_df)

	return pd.concat(centered_dfs, ignore_index=True)

def center_days_since_eq(df):
	"""
	Adds a column 'days_since_eq' to the dataframe, representing the number of days 
	before or after the reference earthquake date (2004-12-26).

	Each date in the 'date' column is converted to an integer count of days relative 
	to the earthquake. Negative values indicate days before the earthquake, and 
	positive values indicate days after.

	Args:
		df (pd.DataFrame): DataFrame that must contain a 'date' column of type datetime 
		                   or convertible to datetime.

	Returns:
		pd.DataFrame: The same DataFrame with a new column 'days_since_eq' added.
	"""
	# Define the earthquake reference date
	reference_date = datetime(2004, 12, 26)
	
	# Ensure 'date' column is in datetime format
	
	# Calculate days relative to the reference date
	df['days_since_eq'] = (df['date'] - reference_date).dt.days
	
	return df


def main():
	df = pd.read_pickle(r'output\preprocessed.pkl')
	print(get_reference_value(df,'ARAU','d_north_mm'))

if __name__ == "__main__":
    main()

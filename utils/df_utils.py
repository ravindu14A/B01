import pandas as pd

def get_all_station_names(df):
	"""Returns a list of unique station names."""
	return df['station'].unique()

# utils.py (Utility functions)
def get_closest_values(df, target_date, columns=None, direction='both', date_col='date'):
	"""
	Returns the row values for specified columns closest to the given date.

	Args:
		df (pd.DataFrame): DataFrame containing the data.
		target_date (datetime or str): The date to search around.
		columns (list or None): List of column names to return. Defaults to None (returns all columns).
		direction (str): 'before', 'after', or 'both' (default).
		date_col (str): Name of the column containing dates (default: 'date').

	Returns:
		dict: Dictionary of column values from the closest row.
	"""
	# Convert target_date to datetime if it's a string
	target_date = pd.to_datetime(target_date)

	# Filter based on direction
	if direction == 'before':
		filtered = df[df[date_col] <= target_date]
	elif direction == 'after':
		filtered = df[df[date_col] >= target_date]
	else:
		filtered = df.copy()

	if filtered.empty:
		return None  # or raise ValueError("No data in the specified direction.")

	# Find row with minimum time difference
	closest_idx = (filtered[date_col] - target_date).abs().idxmin()
	closest_row = filtered.loc[closest_idx]

	# Set default value for columns if None is provided (return all columns)
	if columns is None:
		columns = df.columns

	# Return only desired columns as a dictionary
	return {col: closest_row[col] for col in columns}

def get_stations_with_min_data(df, min_points_pre=50, min_points_after=300):
	"""
	Returns a list of station names that have at least `min_points_pre` data points before the earthquake
	and at least `min_points_after` data points after the earthquake.

	Args:
		df (pd.DataFrame): DataFrame containing 'station' and 'days_since_eq' columns.
		min_points_pre (int): Minimum required data points before the earthquake.
		min_points_after (int): Minimum required data points after the earthquake.

	Returns:
		list: Station names meeting the criteria.
	"""
	useful_stations = []
	for station in get_all_station_names(df):
		subset = df[df['station'] == station]
		n_pre = (subset['days_since_eq'] < 0).sum()
		n_post = (subset['days_since_eq'] > 0).sum()
		if n_pre >= min_points_pre and n_post >= min_points_after:
			useful_stations.append(station)
	return useful_stations

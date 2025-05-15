import pandas as pd

def get_closest_values(df, target_date, columns, direction='both', date_col='date'):
	"""
	Returns the row values for specified columns closest to the given date.

	Args:
		df (pd.DataFrame): DataFrame containing the data.
		target_date (datetime or str): The date to search around.
		columns (list): List of column names to return.
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

	# Return only desired columns as a dictionary
	return {col: closest_row[col] for col in columns}



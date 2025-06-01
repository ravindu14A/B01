import pandas as pd

def drop_days_after_events(df, event_days, N):
	"""
	Drop data points from the DataFrame that are within N days after each event day.
	
	Parameters:
	- df: pandas DataFrame with a 'days_since_eq' column
	- event_days: list of days (floats or ints) representing event days (in days_since_eq)
	- N: number of days to exclude after each event (inclusive)
	
	Returns:
	- Filtered DataFrame
	"""
	mask = pd.Series([True] * len(df), index=df.index)

	for day in event_days:
		mask &= ~((df['days_since_eq'] >= day) & (df['days_since_eq'] <= day + N))

	return df[mask].copy()

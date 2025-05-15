import pandas as pd
from analysis import fitting
import numpy as np


def remove_extreme_outliers(df: pd.DataFrame, column_name: str, threshold: float = 10000):
    """
    Removes extreme outliers for a specified column that exceed a large threshold value.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to check for outliers
    column_name : str
        Name of the column to check for extreme values
    threshold : float, optional
        Threshold value to use for outlier detection (default: 10000)
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with extreme outliers removed
        
    Notes:
    ------
    This function checks for values where the absolute magnitude exceeds the threshold
    and removes those rows from the dataframe. It's meant as a simple sanity check
    to catch unreasonable values before more sophisticated outlier detection.
    """
    # Ensure the column exists
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in dataframe")
        return df
    
    # Find rows with values exceeding the threshold (in either direction)
    extreme_mask = df[column_name].abs() > threshold
    
    # Count outliers found
    outlier_count = extreme_mask.sum()
    
    if outlier_count > 0:
        # Log information about removed outliers
        print(f"Removing {outlier_count} extreme outliers from column '{column_name}' (|value| > {threshold})")
        
        # Optionally, examine the outliers before removing
        # print(df.loc[extreme_mask, ['station', 'date', column_name]])
        
        # Remove the outliers
        df = df.loc[~extreme_mask].copy()
    
    return df

def remove_pre_eq_outliers_en_station(df: pd.DataFrame, station_name: str):
	"""
	Removes pre-earthquake outliers for a given station based on linear fits to east and north data.
	Uses residuals and a robust MAD-based threshold for outlier detection.
	"""
	# Find start of data for station
	start_day = df[df['station'] == station_name]['days_since_eq'].min()

	# Fit east and north trends from start to earthquake (day 0)
	k_e, c_e = fitting.fit_station_linear_trend(df, station_name, "d_east_mm", start_day=start_day, end_day=0)
	k_n, c_n = fitting.fit_station_linear_trend(df, station_name, "d_north_mm", start_day=start_day, end_day=0)

	# Filter pre-earthquake data
	mask = (df['station'] == station_name) & (df['days_since_eq'] < 0)
	df_station = df.loc[mask].copy()

	# Predict expected values from linear model
	df_station['expected_east'] = k_e * df_station['days_since_eq'] + c_e
	df_station['expected_north'] = k_n * df_station['days_since_eq'] + c_n

	# Compute residuals
	df_station['residual_east'] = df_station['d_east_mm'] - df_station['expected_east']
	df_station['residual_north'] = df_station['d_north_mm'] - df_station['expected_north']

	# Combine into total residual magnitude
	df_station['residual_norm'] = np.sqrt(df_station['residual_east']**2 + df_station['residual_north']**2)

	# Compute MAD and robust sigma estimate
	mad = np.median(np.abs(df_station['residual_norm'] - np.median(df_station['residual_norm'])))
	robust_sigma = 1.4826 * mad

	# Threshold for outlier detection
	threshold = 3 * robust_sigma

	# Identify outliers
	outlier_mask = df_station['residual_norm'] > threshold

	# Optionally print or log removed points
	# print(df_station[outlier_mask][['date', 'residual_norm']])

	# Remove outliers from original DataFrame
	df.drop(df_station[outlier_mask].index, inplace=True)

	return df

from analysis.fitting import *
def remove_post_eq_outliers_pc1_station(df: pd.DataFrame, station_name: str, v: float, rotation_matrix: np.ndarray):
    """
    Removes post-earthquake outliers for a given station based on exponential decay fits to PC1 data.
    Uses known velocity parameter v and rotates EN covariance to PC space for outlier detection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data with station measurements
    station_name : str
        Name of the station to process
    v : float
        Known velocity parameter for the exponential decay model
    rotation_matrix : np.ndarray
        PCA eigenvectors matrix to rotate from EN to PC coordinates
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with outliers removed
    """
    # Filter post-earthquake data for the station
    mask = (df['station'] == station_name) & (df['days_since_eq'] >= 0)
    df_station = df.loc[mask].copy()
    
    if df_station.empty:
        return df  # No data to process
    
    # Find end day for fitting
    end_day = df_station['days_since_eq'].max()
    
    # Fit PC1 with exponential decay model using known v
    # Assuming PC1 column exists - adjust as needed
    popt, _ = fitting.fit_station_exponential_decay(
        df, 
        station_name, 
        start_day=0, 
        end_day=end_day, 
        column_name="pc1", 
        v=v
    )
    
    if popt is None:
        return df  # Fitting failed, return original dataframe
    
    # Extract model parameters
    c1, m1, c2, m2, d = popt
    
    # Predict expected values from exponential model
    df_station['expected_pc1'] = model_with_known_v(
        df_station['days_since_eq'], 
        c1, m1, c2, m2, d, v
    )
    
    # Compute residuals for PC1
    df_station['residual_pc1'] = df_station['pc1'] - df_station['expected_pc1']
    
    # We need to rotate the covariance from EN to PC to properly evaluate outliers
    # First, extract the EN covariance (typically this would be from overall station data)
    # For this example, we'll compute it from the pre-earthquake data
    pre_eq_mask = (df['station'] == station_name) & (df['days_since_eq'] < 0)
    pre_eq_data = df.loc[pre_eq_mask, ['d_east_mm', 'd_north_mm']]
    
    if len(pre_eq_data) < 2:
        # Not enough data for covariance estimation
        # Use identity covariance as fallback
        cov_en = np.eye(2)
    else:
        cov_en = np.cov(pre_eq_data['d_east_mm'], pre_eq_data['d_north_mm'])
    
    # Rotate the EN covariance to PC space
    cov_pc = rotation_matrix @ cov_en @ rotation_matrix.T

    
    # Extract the variance of PC1 (first diagonal element of the rotated covariance)
    var_pc1 = cov_pc[0, 0]
    sigma_pc1 = np.sqrt(var_pc1)
    
    # Set threshold for outlier detection (assuming normalized residuals)
    threshold = 3 * sigma_pc1
    
    # Identify outliers based on PC1 residuals
    outlier_mask = np.abs(df_station['residual_pc1']) > threshold
    
    # Optionally print or log removed points
    # print(f"Removing {outlier_mask.sum()} outliers from station {station_name}")
    # print(df_station[outlier_mask][['date', 'days_since_eq', 'residual_pc1']])
    
    # Remove outliers from original DataFrame
    df.drop(df_station[outlier_mask].index, inplace=True)
    
    return df



from scipy.stats import chi2
def depr_mahalanobis(df: pd.DataFrame, station_name: str):
	"""
	Removes the outliers pre-eq for a certain station based on east, north linear interpolations
	"""
	start_day = df[df['station'] == station_name]['days_since_eq'].min()


	k_e, c_e = fitting.fit_station_linear_trend(df, station_name, "d_east_mm", start_day = start_day, end_day = 0)
	k_n, c_n = fitting.fit_station_linear_trend(df, station_name, "d_north_mm", start_day = start_day, end_day = 0)

	threshold = chi2.ppf(0.95, df=2)

	# Filter pre-earthquake data for this station
	mask = (df['station'] == station_name) & (df['days_since_eq'] < 0)
	df_station = df.loc[mask].copy()

	# Predict expected east/north based on linear trend
	df_station['expected_east'] = k_e * df_station['days_since_eq'] + c_e
	df_station['expected_north'] = k_n * df_station['days_since_eq'] + c_n

	def mahalanobis_squared(row):
		# Extract observed and expected vectors
		x = np.array([row['d_east_mm'], row['d_north_mm']])
		mu = np.array([row['expected_east'], row['expected_north']])

		# Extract 2x2 East-North covariance matrix
		cov = np.array(row['enu_covariance_mm2'])[:2, :2]

		# Invert covariance matrix safely
		try:
			inv_cov = np.linalg.inv(cov)
		except np.linalg.LinAlgError:
			return np.nan  # treat as undefined if covariance not invertible

		diff = x - mu
		return float(diff.T @ inv_cov @ diff)

	# Compute Mahalanobis distances
	df_station['mahal_dist2'] = df_station.apply(mahalanobis_squared, axis=1)
	print(df_station['mahal_dist2'])
	# Mark outliers (distance squared > threshold)
	outlier_mask = df_station['mahal_dist2'] > threshold
	print(df_station[outlier_mask]["date"])
	from datetime import datetime
	print(df_station[df['date'] > datetime(2004,1,1)][['date','d_north_mm', 'd_east_mm','enu_covariance_mm2']])

	# Drop outliers from original dataframe
	df.drop(df_station[outlier_mask].index, inplace=True)

	return df
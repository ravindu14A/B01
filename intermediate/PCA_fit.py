import numpy as np
import pandas as pd
from datetime import datetime

def compute_and_apply_pca_ne_station(df, station_name, max_day=365):
	"""
	Computes PCA on North-East displacement data for a station within a time window after the earthquake
	and applies the projection to the full data for that station.

	Includes a synthetic point (0, 0) at day 0 to anchor PCA to the known zero-displacement at the earthquake.

	Args:
		df (pd.DataFrame): DataFrame with 'station', 'days_since_eq', 'd_north_mm', 'd_east_mm' columns.
		station_name (str): The station to process.
		max_day (int): Maximum days after the earthquake to include in the PCA fitting window (default: 365).

	Returns:
		tuple:
			- df (pd.DataFrame): Original DataFrame with added 'pc1' and 'pc2' columns for the station.
			- eigenvectors (np.ndarray): 2×2 matrix of PCA eigenvectors.
	"""
	# Filter data within the PCA fitting window
	mask_fit = (
		(df['station'] == station_name) &
		(df['days_since_eq'] >= 0) &
		(df['days_since_eq'] <= max_day)
	)
	pca_fit_df = df.loc[mask_fit, ['d_north_mm', 'd_east_mm']].copy()
	
	if pca_fit_df.shape[0] <= 1: #!!! idk if this is optimal
		print(f"Warning: Not enough data for PCA on station '{station_name}'. Skipping.")
		return df, None

	# Add the known zero point at the earthquake date
	pca_fit_df.loc[-1] = [0.0, 0.0]  # Add zero displacement point manually
	pca_fit_df.index = range(len(pca_fit_df))  # Reindex cleanly

	# Compute the covariance matrix
	X = pca_fit_df[['d_north_mm', 'd_east_mm']].values
	cov_matrix = (X.T @ X) / (X.shape[0] - 1)

	# Eigen decomposition and sorting
	eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
	sorted_indices = np.argsort(eigenvalues)[::-1]
	eigenvectors = eigenvectors[:, sorted_indices]

	# Project the full station data using the PCA basis
	mask_station = df['station'] == station_name
	X_station = df.loc[mask_station, ['d_north_mm', 'd_east_mm']].values
	components = X_station @ eigenvectors

	# Store projections
	df.loc[mask_station, 'pc1'] = components[:, 0]
	df.loc[mask_station, 'pc2'] = components[:, 1]
	
	return df, eigenvectors

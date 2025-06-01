import numpy as np
import pandas as pd
from datetime import datetime

def compute_and_apply_pca_ne_station(df, station_name, max_day=365):
	"""
	Compute PCA on d_north_mm and d_east_mm for a station within a time window after the earthquake.
	Then rotate all NE covariance matrices into PC space and store the result in a new column.

	Adds:
		- pc1, pc2: projection of NE displacement onto PCA basis
		- pca_covariance_pc_space: 2×2 rotated covariance matrix for each point (in cm²)
	"""
	import numpy as np
	import pandas as pd

	# Filter for PCA fitting
	mask_fit = (
		(df['station'] == station_name) &
		(df['days_since_eq'] >= 0) &
		(df['days_since_eq'] <= max_day)
	)
	pca_fit_df = df.loc[mask_fit, ['d_north_mm', 'd_east_mm']].copy()

	if pca_fit_df.shape[0] <= 1:
		print(f"Warning: Not enough data for PCA on station '{station_name}'. Skipping.")
		return df, None

	# Add known zero point
	pca_fit_df.loc[-1] = [0.0, 0.0]
	pca_fit_df.index = range(len(pca_fit_df))

	# PCA eigenvectors
	X = pca_fit_df.values
	cov_matrix = (X.T @ X) / (X.shape[0] - 1)
	eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
	eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]  # sort descending

	# Project full station data into PC space
	mask_station = df['station'] == station_name
	X_station = df.loc[mask_station, ['d_north_mm', 'd_east_mm']].values
	components = X_station @ eigenvectors
	df.loc[mask_station, 'pc1'] = components[:, 0]
	df.loc[mask_station, 'pc2'] = components[:, 1]

	# Rotate each row's covariance into PC-space
	def rotate_cov(row):
		if isinstance(row['enu_covariance_cm2'], np.ndarray):
			# Extract NE 2×2 from ENU covariance
			cov_ne_cm2 = row['enu_covariance_cm2'][:2, :2]
			cov_pc = eigenvectors.T @ cov_ne_cm2 @ eigenvectors
			return cov_pc
		else:
			return np.full((2, 2), np.nan)

	df.loc[mask_station, 'pca_covariance_pc_space'] = df.loc[mask_station].apply(rotate_cov, axis=1)

	return df, eigenvectors

import pandas as pd
import matplotlib.pyplot as plt

from analysis import velocity_correction
#
# df = pd.read_pickle(r"output/preprocessed.pkl")
# df = velocity_correction.apply_velocity_correction(df)
#
#
# pd.to_pickle(df,r"output\velocity_corrected.pkl")
df = pd.read_pickle(r'output\velocity_corrected.pkl')
print(df[df['station']=="PHUK"][["date"]].to_numpy())


def plot_displacement_for_station(df, station_name):
	# Filter DataFrame by the station name
	station_data = df[df['station'] == station_name].copy()

	# Create subplots
	fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

	# North displacement
	axs[0].plot(station_data['date'], station_data['d_north_mm'], color='b', linewidth=1)
	axs[0].set_ylabel('North (mm)')
	axs[0].set_title(f"Displacement for Station {station_name}")
	axs[0].grid(True)

	# East displacement
	axs[1].plot(station_data['date'], station_data['d_east_mm'], color='g', linewidth=1)
	axs[1].set_ylabel('East (mm)')
	axs[1].grid(True)

	# Height displacement
	axs[2].plot(station_data['date'], station_data['d_up_mm'], color='r', linewidth=1)
	axs[2].set_ylabel('Height (mm)')
	axs[2].set_xlabel('Date')
	axs[2].grid(True)

	plt.tight_layout()
	plt.show()

def scatterplot(df, station_name):
	# Filter and copy data
	station_data = df[df['station'] == station_name].copy()

	# Convert to datetime and extract year
	station_data['date'] = pd.to_datetime(station_data['date'])
	station_data['year'] = station_data['date'].dt.year + station_data['date'].dt.dayofyear / 365.25

	# Create scatter plot with color gradient by year
	plt.figure(figsize=(8, 6))
	sc = plt.scatter(
		station_data['d_north_mm'],
		station_data['d_east_mm'],
		c=station_data['year'],
		cmap='inferno',  # or 'plasma', 'cool', etc.
		s=10,
		alpha=0.8
	)
	plt.xlabel('North Displacement (mm)')
	plt.ylabel('East Displacement (mm)')
	plt.title(f'Displacement Scatter Plot for Station {station_name}')
	cbar = plt.colorbar(sc)
	cbar.set_label('Year')
	plt.grid(True)
	plt.tight_layout()
	plt.show()




# Example usage
plot_displacement_for_station(df, "PHUK")  # Replace "NTUS" with your desired station name
scatterplot(df,"PHUK")



#PCA TRANSFORMATION PLOT AND DATA STORED  AND COMMENTED BELOW!!!!!!!!!!!!!

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
#
#
# def load_data(path="output/velocity_corrected.pkl"):
#     df = pd.read_pickle(path)
#     return df
#
#
#
# def get_station_data(df, station_name):
#     sd = df[df['station'] == station_name].copy()
#     sd['date'] = pd.to_datetime(sd['date'])
#     sd['year'] = sd['date'].dt.year + sd['date'].dt.dayofyear / 365.25
#     return sd
#
#
#
# def plot_displacement_for_station(station_data, station_name):
#     fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#
#     axs[0].plot(station_data['date'], station_data['d_north_mm'], linewidth=1)
#     axs[0].set_ylabel('North (mm)')
#     axs[0].set_title(f"Displacement Time Series for {station_name}")
#     axs[0].grid(True)
#
#     axs[1].plot(station_data['date'], station_data['d_east_mm'], linewidth=1, color='orange')
#     axs[1].set_ylabel('East (mm)')
#     axs[1].grid(True)
#
#     axs[2].plot(station_data['date'], station_data['d_up_mm'], linewidth=1, color='green')
#     axs[2].set_ylabel('Up (mm)')
#     axs[2].set_xlabel('Date')
#     axs[2].grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
#
#
# def scatterplot_ne(station_data, station_name):
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(
#         station_data['d_north_mm'],
#         station_data['d_east_mm'],
#         c=station_data['year'],
#         cmap='inferno',
#         s=15,
#         alpha=0.8
#     )
#     plt.xlabel('North Displacement (mm)')
#     plt.ylabel('East Displacement (mm)')
#     plt.title(f'N–E Scatter for {station_name}')
#     cb = plt.colorbar(sc)
#     cb.set_label('Year')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
#
# def compute_and_plot_pca_ne(station_data, station_name):
#
#     X = station_data[['d_north_mm', 'd_east_mm']].values
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#
#
#     station_data['pc1'] = X_pca[:, 0]
#     station_data['pc2'] = X_pca[:, 1]
#
#
#     evr = pca.explained_variance_ratio_
#     print(f"Explained variance ratio: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}")
#
#
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(
#         station_data['pc1'], station_data['pc2'],
#         c=station_data['year'],
#         cmap='inferno',
#         s=15,
#         alpha=0.8
#     )
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.title(f'PCA of N–E Offsets for {station_name}')
#     cb = plt.colorbar(sc)
#     cb.set_label('Year')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
#     return station_data
#
#
#
# if __name__ == "__main__":
#     station = "PHUK"
#
#     df = load_data()
#     station_df = get_station_data(df, station)
#
#
#     plot_displacement_for_station(station_df, station)
#     scatterplot_ne(station_df, station)
#
#
#     station_df = compute_and_plot_pca_ne(station_df, station)
#

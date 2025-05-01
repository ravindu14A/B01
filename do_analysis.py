import pandas as pd
import matplotlib.pyplot as plt

from analysis import velocity_correction


# df = velocity_correction.apply_velocity_correction(df)
# pd.to_pickle(df,r"output\velocity_corrected.py")

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
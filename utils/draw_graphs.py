import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
import contextily as ctx
from shapely.geometry import box
from analysis.fitting import model_with_known_v, model_without_v_term

#from analysis.depr_curve_fit import model_with_known_v, model_without_v_term
import numpy as np
df = pd.read_pickle(r'output\velocity_corrected.pkl')

#print(df[df['station']=="PHUK"][["date"]].to_numpy())


def plot_displacement_for_station(df, station_name): # dispalcement north, east, height in mm (v corrected depends on given df)
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

	df_earthquakes = pd.read_pickle(r'raw_data\earthquakes_records')


	vline_dates = df_earthquakes['date']
	# Optional: Add vertical lines
	if not vline_dates.empty:
		for ax in axs:
			for vline_date in vline_dates:
				ax.axvline(vline_date, color='k', linestyle='--', linewidth=1)


	# Set x-axis to display years
	axs[2].xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
	axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format ticks as year



	plt.tight_layout()
	plt.show()

def scatterplot_ne(df, station_name): # east vs west 
	# Filter and copy data
	station_data = df[df['station'] == station_name].copy()

	# Convert to datetime and extract year
	station_data['date'] = pd.to_datetime(station_data['date'])
	station_data['year'] = station_data['date'].dt.year + station_data['date'].dt.dayofyear / 365.25

	# Create scatter plot with color gradient by year
	plt.figure(figsize=(8, 6))
	sc = plt.scatter(
		station_data['d_east_mm'],
		station_data['d_north_mm'],
		c=station_data['year'],
		cmap='inferno',  # or 'plasma', 'cool', etc.
		s=10,
		alpha=0.8
	)
	plt.xlabel('East Displacement (mm)')
	plt.ylabel('North Displacement (mm)')
	plt.title(f'Displacement Scatter Plot for Station {station_name}')
	cbar = plt.colorbar(sc)
	cbar.set_label('Year')
	plt.grid(True)
	plt.tight_layout()
	plt.show()



#print(ctx.providers.keys())  # List top-level providers

def plot_stations_on_map(df: pd.DataFrame, station_names: list):
	all_stations_first = df.sort_values("date").drop_duplicates("station")
	
	# Create GeoDataFrame of all stations (for consistent extent)
	gdf_all = gpd.GeoDataFrame(
		all_stations_first,
		geometry=gpd.points_from_xy(all_stations_first.lon, all_stations_first.lat),
		crs="EPSG:4326"
	).to_crs(epsg=3857)

	# Subset GeoDataFrame based on given station names
	gdf_subset = gdf_all[gdf_all["station"].isin(station_names)]

	# Plotting
	fig, ax = plt.subplots(figsize=(10, 10))
	gdf_all.plot(ax=ax, color="lightgray", markersize=10, label="All Stations")
	gdf_subset.plot(ax=ax, color="red", markersize=30, label="Selected Stations")

	# Label selected stations
	for x, y, label in zip(gdf_subset.geometry.x, gdf_subset.geometry.y, gdf_subset["station"]):
		ax.text(x, y, label, fontsize=8, ha='right')

	# Set extent to full bounds with some padding
	buffer = 5000  # in meters
	bounds = gdf_all.total_bounds  # minx, miny, maxx, maxy
	# ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
	# ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
	# Define bounding box in lat/lon
	min_lon, max_lon = 90, 120
	min_lat, max_lat = -20, 20
	# Convert to GeoDataFrame and reproject to Web Mercator
	bbox = gpd.GeoSeries([box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326").to_crs(epsg=3857)
	minx, miny, maxx, maxy = bbox.total_bounds

	# Set map limits
	ax.set_xlim(minx, maxx)
	ax.set_ylim(miny, maxy)

	# Add basemap
	ctx.add_basemap(ax, crs=gdf_all.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

	ax.set_title("Station Locations")
	ax.axis("off")
	ax.legend()
	plt.tight_layout()
	plt.show()
# Example usage

def plot_earthquakes_on_map(df: pd.DataFrame):
	# Ensure 'date' is datetime
	df = df.copy()
	df["date"] = pd.to_datetime(df["date"])
	df["year_short"] = df["date"].dt.year % 100  # Get last two digits

	# Create GeoDataFrame
	gdf_eq = gpd.GeoDataFrame(
		df,
		geometry=gpd.points_from_xy(df.lon, df.lat),
		crs="EPSG:4326"
	).to_crs(epsg=3857)

	# Plotting
	fig, ax = plt.subplots(figsize=(10, 10))

	# Normalize magnitude for color mapping (green to red)
	norm = plt.Normalize(df["mag"].min(), df["mag"].max())
	cmap = plt.cm.get_cmap("RdYlGn_r")

	# Plot earthquake points
	gdf_eq.plot(
		ax=ax,
		markersize=20,
		color=[cmap(norm(m)) for m in df["mag"]],
		alpha=0.8,
	)

	# Add labels: "Mag 'YY"
	for x, y, mag, yy in zip(gdf_eq.geometry.x, gdf_eq.geometry.y, df["mag"], df["year_short"]):
		ax.text(x, y, f"{mag:.1f} '{yy:02}", fontsize=7, ha='left', va='center')

	# Set map bounds (manual or based on data)
	min_lon, max_lon = 90, 120
	min_lat, max_lat = -20, 20
	bbox = gpd.GeoSeries([box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326").to_crs(epsg=3857)
	minx, miny, maxx, maxy = bbox.total_bounds
	ax.set_xlim(minx, maxx)
	ax.set_ylim(miny, maxy)

	# Add basemap
	ctx.add_basemap(ax, crs=gdf_eq.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

	ax.set_title("Earthquake Locations with Magnitudes and Years")
	ax.axis("off")
	plt.tight_layout()
	plt.show()


def plot_column_for_station(df, station_name, column, v=None, popt=None):
	"""
	Plots time series data for a specific station and column, with optional model fitting.
	
	Parameters:
	-----------
	df : pandas.DataFrame
		DataFrame containing the station data
	station_name : str
		Name of the station to plot
	column : str
		Column name to plot (e.g., 'd_east_mm', 'pc1')
	v : float, optional
		Known velocity parameter for model fitting
	popt : tuple, optional
		Fitted parameters for the exponential decay model (c1, m1, c2, m2, d)
		
	Notes:
	------
	The function assumes 'days_since_eq' is relative to the 2004-12-26 earthquake.
	"""
	# Filter DataFrame by station
	station_data = df[df['station'] == station_name].copy()
	station_data['date'] = pd.to_datetime(station_data['date'])
	
	# Define earthquake date for reference
	quake_date = pd.Timestamp("2004-12-26")
	
	# Plot setup
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(station_data['date'], station_data[column], color='b', linewidth=1, label='Observed')
	ax.scatter(station_data['date'], station_data[column], color='red', s=10, label='Data Points')
	
	# Plot model if parameters provided
	if popt is not None:
		# Generate model timeline (100 years from earthquake)
		end_model_date = quake_date + pd.DateOffset(years=100)
		 
		# Generate daily dates from earthquake date to end date
		model_dates = pd.date_range(start=quake_date, end=end_model_date, freq='D')
		
		# Calculate days_since_eq for model dates
		days_since_eq = np.array([(date - quake_date).days for date in model_dates])
		
		# Generate model values
		if v is not None:
			y_model = model_with_known_v(days_since_eq, *popt, v)
		else:
			y_model = model_without_v_term(days_since_eq, *popt)
		
		# Plot model
		ax.plot(model_dates, y_model, color='orange', linestyle='--', linewidth=2, label='Fitted Model (post-2004)')
	
	# Set Y-axis limit
	bottom_limit = min(-500, station_data[column].min() * 1.1)  # Dynamic lower limit
	ax.set_ylim(bottom=bottom_limit, top=station_data[column].max() * 1.1)
	ax.set_xlim(-500, 365*300)
	# Labels and formatting
	ax.set_ylabel(column.replace('d_', '').replace('_mm', '').capitalize() + ' (mm)')
	ax.set_title(f"{column} Displacement for Station {station_name}")
	ax.grid(True)
	
	# Earthquake vertical lines
	try:
		df_earthquakes = pd.read_pickle(r'raw_data\earthquakes_records')
		if not df_earthquakes.empty:
			for date in df_earthquakes['date']:
				ax.axvline(date, color='k', linestyle='--', linewidth=1)
	except Exception as e:
		print(f"Could not load earthquake data: {e}")
	
	# Format x-axis as years
	ax.xaxis.set_major_locator(mdates.YearLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
	ax.set_xlabel('Date')
	ax.legend()
	
	plt.tight_layout()
	plt.show()


def main():
	stationname = "PHUK"
	plot_displacement_for_station(df, stationname)  # Replace "NTUS" with your desired station name
	scatterplot_ne(df,stationname)
	plot_stations_on_map(df,['ARAU','PHUK','PHKT'])

	df_earthquakes = pd.read_pickle(r'raw_data\earthquakes_records')

	plot_earthquakes_on_map(df_earthquakes)

if __name__ == '__main__':
	main()
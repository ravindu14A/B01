import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar

# Load station coordinates
station_coord_raw = pd.read_csv(r'C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\SE_Asia\SE_Asia_stations_with_20plus.csv')
station_ident = station_coord_raw['File']
station_latitude = station_coord_raw['Average Latitude'] * (180 / np.pi)
station_longitude = station_coord_raw['Average Longitude'] * (180 / np.pi)

# Combine station data into a list of tuples
station_coord_proc = list(zip(station_ident, station_latitude, station_longitude))

# Function to create a precise map with station locations
def plot_stations(station_data):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Mercator()}, dpi=1600)

    # Add high-resolution map features
    ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='--', edgecolor='black')

    # Plot station locations with labels
    for name, lat, lon in station_data:
        ax.plot(lon, lat, marker='^', color='red', markersize=5, transform=ccrs.PlateCarree(), label='Station')
        ax.text(lon + 0.01, lat + 0.01, name, fontsize=4, transform=ccrs.PlateCarree())  # Add station name

    # Add Phuket as a city
    phuket_lat, phuket_lon = 7.8804, 98.3923
    ax.plot(phuket_lon, phuket_lat, marker='o', color='blue', markersize=6, transform=ccrs.PlateCarree(),
            label='Phuket')
    ax.text(phuket_lon + 0.01, phuket_lat + 0.01, "Phuket", fontsize=9, fontweight='bold', color='blue',
            transform=ccrs.PlateCarree())

    # Add refined gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Set precise plot extent based on station locations
    lon_min, lon_max = min(station_longitude) - 1, max(station_longitude) + 1
    lat_min, lat_max = min(station_latitude) - 1, max(station_latitude) + 1
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.set_title("Data Gathering Stations with 20+ datapoints in SE Asia (1999-2025)", fontsize=14, pad=20)

    # Add the scale bar (1 degree ≈ 111 km at the equator)
    ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor='black', facecolor='lightgray')
    scalebar = ScaleBar(1, location='lower right', length_fraction=0.1)  # Adjust the length_fraction to your liking
    ax.add_artist(scalebar)

    # Ensure the figure background is visible
    fig.patch.set_facecolor('white')

    return fig, ax

# Main script
if __name__ == '__main__':
    combined_fig, combined_ax = plot_stations(station_coord_proc)
    plt.show()

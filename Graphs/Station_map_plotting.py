import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib
import os
from matplotlib.lines import Line2D
import geopandas as gpd

# Use a non-interactive backend and default font
matplotlib.use('Agg')

# Load station coordinates
station_coord_raw = pd.read_csv(r'C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\SE_Asia\SE_Asia_avg_coordinates.csv')
station_ident = station_coord_raw['File']
station_latitude = station_coord_raw['Average Latitude'] * (180 / np.pi)
station_longitude = station_coord_raw['Average Longitude'] * (180 / np.pi)

station_coord_proc = list(zip(station_ident, station_latitude, station_longitude))

# Major cities
major_cities = [
    ("Kuala Lumpur", 3.1390, 101.6869),
    ("Jakarta", -6.2088, 106.8456),
    ("Ho Chi Minh City", 10.7769, 106.7009),
    ("Manila", 14.5995, 120.9842),
    ("Hanoi", 21.0285, 105.8544),
    ("Singapore", 1.3521, 103.8198),
    ("Phnom Penh", 11.5564, 104.9282),
]

# Stations to highlight
highlight_stations = {
    "THAILAND": {"PHKT", "PHUK", "ARAU"},
    "MALAYSIA": {"ARAU", "KUAL", "GETI", "USMP"}
}
highlight_set = set().union(*highlight_stations.values())

def plot_stations(station_data):
    fig, ax = plt.subplots(figsize=(24, 6), subplot_kw={'projection': ccrs.Mercator()}, dpi=1200)

    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')

    # Extract station latitudes and longitudes
    lats = [lat for _, lat, _ in station_data]
    lons = [lon for _, _, lon in station_data]

    # Set the extent based on stations
    lon_min, lon_max = min(lons) - 2, max(lons) + 2
    lat_min, lat_max = min(lats) - 2, max(lats) + 2
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Plot tectonic plate boundaries
    plate_path = r"C:\Users\nicov\Downloads\PB2002_boundaries.json"
    plate_boundaries = gpd.read_file(plate_path)
    plate_boundaries.plot(ax=ax, edgecolor='orangered', linewidth=0.8,
                          facecolor='none', transform=ccrs.PlateCarree(), zorder=3)

    # Plot stations
    for ident, lat, lon in station_data:
        if ident in highlight_set:
            ax.plot(lon, lat, marker='d', color='yellow', markersize=4, transform=ccrs.PlateCarree(), zorder=5)
        else:
            ax.plot(lon, lat, marker='^', color='purple', markersize=4, transform=ccrs.PlateCarree(), zorder=4)

    # Plot major cities
    for city, lat, lon in major_cities:
        if (lon_min < lon < lon_max) and (lat_min < lat < lat_max):
            ax.plot(lon, lat, marker='o', color='black', markersize=5, transform=ccrs.PlateCarree())
            ax.text(lon + 0.2, lat + 0.2, city,
                    fontsize=7, fontweight='semibold', color='darkslategray',
                    transform=ccrs.PlateCarree(), zorder=6)

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': 'black'}
    gl.ylabel_style = {'size': 8, 'color': 'black'}

    # Add legend
    station_marker = Line2D([0], [0], marker='^', color='purple', linestyle='None', markersize=6, label='Ground Station')
    highlight_marker = Line2D([0], [0], marker='d', color='yellow', linestyle='None', markersize=8, label='Stations selected for this study')
    boundary_marker = Line2D([0], [0], color='orangered', linewidth=1, linestyle='-', label='Tectonic Plate Boundaries')
    ax.legend(handles=[station_marker, highlight_marker, boundary_marker], loc='upper right', fontsize=8)

    # Add scale bar
    scalebar = ScaleBar(
        dx=1,
        units='m',
        scale_formatter=lambda value, unit: f'{value/1000:.0f} km',
        location='lower left',
        length_fraction=0.2,
        scale_loc='bottom',
        border_pad=0.5,
        fixed_value=500000,
        fixed_units='m'
    )
    ax.add_artist(scalebar)

    fig.patch.set_facecolor('white')
    return fig, ax

if __name__ == '__main__':
    fig, ax = plot_stations(station_coord_proc)

    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    save_path = os.path.join(downloads_path, "all_datapoints_station_map_highlighted.png")
    fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    print(f"Saved to: {save_path}")

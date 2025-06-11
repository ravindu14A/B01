import matplotlib
matplotlib.use('TkAgg')  # Open in a window

import zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from io import BytesIO

# === 1. File path ===
ZIP_FILE_PATH = r"C:\Users\nicov\Downloads\au_dem_9_1.zip"

# === 2. Jakarta bounds ===
COORDINATE_BOUNDS = {
    'min_lon': 114.55,  # West of the city
    'max_lon': 115.55,  # East of the city
    'min_lat': 4.45,    # South of the city
    'max_lat': 5.45     # North of the city
}
# === 3. Settings ===
MAX_ALTITUDE = 100  # meters
DPI = 400
PLOT_SIZE = (12, 8)

def latlon_to_pixel(lat, lon, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)

try:
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        tif_files = [f for f in z.namelist() if f.endswith('.tif')]
        if not tif_files:
            raise FileNotFoundError("No .tif file found in the ZIP archive.")

        with z.open(tif_files[0]) as tif_file:
            with rasterio.open(BytesIO(tif_file.read())) as src:
                print(f"Opened DEM file. CRS: {src.crs}")

                # Convert bounds
                row_min, col_min = latlon_to_pixel(COORDINATE_BOUNDS['max_lat'],
                                                   COORDINATE_BOUNDS['min_lon'],
                                                   src.transform)
                row_max, col_max = latlon_to_pixel(COORDINATE_BOUNDS['min_lat'],
                                                   COORDINATE_BOUNDS['max_lon'],
                                                   src.transform)

                row_start, row_stop = sorted([row_min, row_max])
                col_start, col_stop = sorted([col_min, col_max])
                window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
                data = src.read(1, window=window)

                # Land elevation map
                land_data = np.clip(data, 0, MAX_ALTITUDE)

                # Sea mask
                sea_mask = data <= 0

                # Aspect ratio
                lat_span = COORDINATE_BOUNDS['max_lat'] - COORDINATE_BOUNDS['min_lat']
                lon_span = COORDINATE_BOUNDS['max_lon'] - COORDINATE_BOUNDS['min_lon']
                aspect_ratio = lat_span / lon_span

                # Estimate meters per pixel
                lat_mean = np.mean([COORDINATE_BOUNDS['min_lat'], COORDINATE_BOUNDS['max_lat']])
                meters_per_deg_lon = 111320 * np.cos(np.radians(lat_mean))
                meters_per_deg_lat = 110574
                dx = abs(src.transform.a) * meters_per_deg_lon
                dy = abs(src.transform.e) * meters_per_deg_lat
                meters_per_pixel = (dx + dy) / 2

                # === Plot ===
                plt.figure("Brunei Elevation Map", figsize=PLOT_SIZE, dpi=DPI)
                manager = plt.get_current_fig_manager()
                manager.window.wm_geometry("+50+50")

                extent = [COORDINATE_BOUNDS['min_lon'], COORDINATE_BOUNDS['max_lon'],
                          COORDINATE_BOUNDS['min_lat'], COORDINATE_BOUNDS['max_lat']]

                # Plot terrain elevation
                img = plt.imshow(land_data, cmap='terrain', vmin=0, vmax=MAX_ALTITUDE,
                                 extent=extent, aspect='auto')

                # Overlay sea as light blue
                plt.imshow(sea_mask, cmap='Blues', alpha=0.4, extent=extent, aspect='auto')

                plt.gca().set_aspect(1.0 / aspect_ratio)

                # Colorbar
                cbar = plt.colorbar(img, label='Elevation (m)', shrink=0.7, pad=0.02)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label('Elevation (m)', fontsize=9)

                # Scale bar
                scalebar = ScaleBar(dx=meters_per_pixel, units='m', length_fraction=0.2,
                                    location='lower right', scale_loc='bottom',
                                    fixed_value=10000, fixed_units='m',
                                    frameon=False, color='black', font_properties={'size': 8})
                plt.gca().add_artist(scalebar)

                # Labels
                plt.xlabel("Longitude", fontsize=10)
                plt.ylabel("Latitude", fontsize=10)
                plt.xticks(fontsize=8, rotation=45)
                plt.yticks(fontsize=8)
                plt.title("Jakarta Elevation Map", fontsize=12, pad=20)
                plt.grid(False)
                plt.tight_layout()
                plt.show()

except Exception as e:
    print(f"Error: {e}")

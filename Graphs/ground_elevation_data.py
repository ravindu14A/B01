import zipfile
import rasterio
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from rasterio.windows import Window
from matplotlib_scalebar.scalebar import ScaleBar

# ========== USER CONFIGURATION ==========
# 1. Define your input file path
ZIP_FILE_PATH = r"C:\Users\nicov\Downloads\au_dem_9_1.zip"  # <<< CHANGE THIS to your actual file path

# 2. Set your geographic area of interest (in decimal degrees)
COORDINATE_BOUNDS = {
    'min_lon': 106.03,  # Western boundary (West Jakarta)
    'max_lon': 107.93,  # Eastern boundary (East Jakarta)
    'min_lat': -6.95,   # Southern boundary (South Jakarta)
    'max_lat': -5.10    # Northern boundary (North Jakarta/Java Sea coast)
}

# 3. Set visualization parameters
MAX_ALTITUDE = 40  # Maximum elevation to display (meters)
DPI = 1000          # Figure resolution


# ========================================

def latlon_to_pixel(lat, lon, transform):
    """Convert geographic coordinates to pixel coordinates"""
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


try:
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        # Find the .tif file in the zip archive
        tif_files = [f for f in z.namelist() if f.endswith('.tif')]
        if not tif_files:
            raise FileNotFoundError("No .tif file found in the zip archive")

        with z.open(tif_files[0]) as tif_file:
            with rasterio.open(BytesIO(tif_file.read())) as src:
                print(f"File opened successfully. CRS: {src.crs}")

                # Convert geographic bounds to pixel coordinates
                try:
                    row_min, col_min = latlon_to_pixel(
                        COORDINATE_BOUNDS['max_lat'],
                        COORDINATE_BOUNDS['min_lon'],
                        src.transform)
                    row_max, col_max = latlon_to_pixel(
                        COORDINATE_BOUNDS['min_lat'],
                        COORDINATE_BOUNDS['max_lon'],
                        src.transform)
                except KeyError as e:
                    raise KeyError(f"Missing coordinate bound: {e}. Please check COORDINATE_BOUNDS dictionary")

                # Create window
                window = Window.from_slices(
                    (row_min, row_max),
                    (col_min, col_max)
                )

                # Read and process
                data = src.read(1, window=window)
                data_clipped = np.clip(data, 0, MAX_ALTITUDE)

                # Calculate approximate meters per pixel
                lon_length = 111320 * np.cos(np.radians(np.mean([COORDINATE_BOUNDS['min_lat'], COORDINATE_BOUNDS['max_lat']])))
                lat_length = 110574
                dx = (src.transform.a * 111320) / lon_length  # meters per pixel in x
                dy = (src.transform.e * 110574) / lat_length  # meters per pixel in y
                meters_per_pixel = (dx + dy) / 2

                # Create figure
                plt.figure(figsize=(12, 8), dpi=DPI)
                img = plt.imshow(data_clipped, cmap='terrain', vmin=0, vmax=MAX_ALTITUDE,
                                extent=[COORDINATE_BOUNDS['min_lon'], COORDINATE_BOUNDS['max_lon'],
                                        COORDINATE_BOUNDS['min_lat'], COORDINATE_BOUNDS['max_lat']])

                # Add colorbar
                cbar = plt.colorbar(label=f'Elevation (0-{MAX_ALTITUDE}m)')
                cbar.ax.yaxis.set_label_position('left')

                # Add scale bar (10km)
                scalebar = ScaleBar(dx=meters_per_pixel, units='m', length_fraction=0.25,
                                  location='lower right', scale_loc='bottom',
                                  fixed_value=10000, fixed_units='m',
                                  frameon=True, color='black')
                plt.gca().add_artist(scalebar)

                # Format axes
                plt.title(f"Elevation Map\n"
                          f"Longitude: {COORDINATE_BOUNDS['min_lon']:.2f}°E to {COORDINATE_BOUNDS['max_lon']:.2f}°E\n"
                          f"Latitude: {COORDINATE_BOUNDS['min_lat']:.2f}°S to {COORDINATE_BOUNDS['max_lat']:.2f}°S")
                plt.xlabel("Longitude (degrees East)")
                plt.ylabel("Latitude (degrees South)")
                plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f°E'))
                plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f°S'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

except FileNotFoundError:
    print(f"Error: File not found at {ZIP_FILE_PATH}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images

import os
import zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO

# === File and region setup ===
ZIP_FILE_PATH = r"C:\Users\nicov\Downloads\au_dem_9_1.zip"
COORDINATE_BOUNDS = {
    'min_lon': 106.6,
    'max_lon': 107.1,
    'min_lat': -6.4,
    'max_lat': -5.9
}
MAX_ALTITUDE = 40
DPI = 150
PLOT_SIZE = (14, 9)

# === Output folder ===
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sea_level_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def latlon_to_pixel(lat, lon, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)

def create_flood_colormap():
    terrain = plt.get_cmap('terrain', 512)
    colors = terrain(np.linspace(0, 1, 512))
    flood_color = np.array([0.4, 0.6, 1.0, 1.0])  # pleasant blue
    for i in range(30):  # Replace first few low elevation colors
        colors[i] = flood_color
    return ListedColormap(colors)

def load_elevation_data():
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        tif_files = [f for f in z.namelist() if f.endswith('.tif')]
        if not tif_files:
            raise FileNotFoundError("No .tif file found in ZIP.")

        with z.open(tif_files[0]) as tif_file:
            with rasterio.open(BytesIO(tif_file.read())) as src:
                row_min, col_min = latlon_to_pixel(COORDINATE_BOUNDS['max_lat'],
                                                   COORDINATE_BOUNDS['min_lon'], src.transform)
                row_max, col_max = latlon_to_pixel(COORDINATE_BOUNDS['min_lat'],
                                                   COORDINATE_BOUNDS['max_lon'], src.transform)
                row_start, row_stop = sorted([row_min, row_max])
                col_start, col_stop = sorted([col_min, col_max])
                window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
                data = src.read(1, window=window).astype(np.float32)
                data[data < 0] = 0
                return data

# === Main batch export loop ===
elevation_data = load_elevation_data()
cmap = create_flood_colormap()

for i in range(0, 101):  # 0 to 1000 cm = 10.0 m
    sea_level = i / 10.0  # meters

    flooded = np.copy(elevation_data)
    flooded[flooded <= sea_level + 1e-2] = 0
    flooded[flooded > sea_level + 1e-2] = elevation_data[flooded > sea_level + 1e-2]

    fig, ax = plt.subplots(figsize=PLOT_SIZE, dpi=DPI)
    img = ax.imshow(flooded, cmap=cmap, vmin=0, vmax=MAX_ALTITUDE,
                    extent=[COORDINATE_BOUNDS['min_lon'], COORDINATE_BOUNDS['max_lon'],
                            COORDINATE_BOUNDS['min_lat'], COORDINATE_BOUNDS['max_lat']],
                    aspect='auto', interpolation='bilinear')

    ax.set_title(f"Sea Level Rise Simulation: {sea_level:.1f} m", fontsize=12)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(False)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"sea_level_{sea_level:.1f}m.png")
    plt.savefig(output_path, dpi=DPI)
    plt.close()

print(f"✅ Finished saving 101 images to: {OUTPUT_DIR}")

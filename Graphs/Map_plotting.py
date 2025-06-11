import multiprocessing
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import pickle
import os
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import MatplotlibDeprecationWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='cartopy')
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# Function to load data from PKL files in multiple directories
def load_pkl_files(directories):
    all_data = []
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                with open(os.path.join(directory, filename), 'rb') as f:
                    data = pickle.load(f)
                    # Filter out bad points with invalid format
                    for point in data:
                        if (
                            isinstance(point, (list, tuple)) and
                            len(point) == 4 and
                            isinstance(point[0], (int, float)) and
                            isinstance(point[1], (int, float)) and
                            isinstance(point[2], (int, float, str))  # allow str to check later
                        ):
                            all_data.append(point)
    return all_data

# Function to compute average location
def compute_average_location(data):
    lats = [point[0] for point in data]
    lons = [point[1] for point in data]
    avg_lat = np.mean(lats)
    avg_lon = np.mean(lons)
    return avg_lat, avg_lon

# Dummy function for parallel processing
def plot_points(data_point):
    return data_point

# Function to combine plots into a single figure
def combine_plots(plot_data):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=600)

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle="--")
    ax.add_feature(cfeature.STATES, linestyle=":")

    # Extract and sanitize magnitudes for colormap
    magnitudes = []
    for sublist in plot_data:
        for point in sublist:
            try:
                magnitudes.append(float(point[2]))
            except (ValueError, TypeError):
                continue

    if not magnitudes:
        raise ValueError("No valid magnitudes found for plotting.")

    norm = Normalize(vmin=min(magnitudes), vmax=max(magnitudes))
    cmap = cm.viridis

    # Plot all points
    for data_point in plot_data:
        for lat, lon, magnitude, time in data_point:
            try:
                mag_float = float(magnitude)
                color = cmap(norm(mag_float))
                ax.plot(lon, lat, marker='o', color=color, markersize=5, transform=ccrs.PlateCarree())
            except (ValueError, TypeError):
                continue

    # Compute and plot average location
    all_points = [point for sublist in plot_data for point in sublist if isinstance(point[0], (int, float)) and isinstance(point[1], (int, float))]
    avg_lat, avg_lon = compute_average_location(all_points)
    ax.plot(avg_lon, avg_lat, marker='*', color='red', markersize=15, label='Average Location', transform=ccrs.PlateCarree())
    ax.legend()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical", label="Magnitude")

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    ax.set_title(f"Earthquakes in Thailand and Malaysia (1999-2025) with Average Location", fontsize=14, pad=20)
    return fig, ax

if __name__ == '__main__':
    # Directories containing PKL files
    pkl_directories = [
        r"C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\Thailand\Raw_pickle",
        r"C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\Malaysia\Raw_pickle"
    ]

    # Load data from PKL files
    coords_try = load_pkl_files(pkl_directories)

    # Split data for multiprocessing
    num_processes = multiprocessing.cpu_count()
    chunk_size = max(1, len(coords_try) // num_processes)
    chunks = [coords_try[i:i + chunk_size] for i in range(0, len(coords_try), chunk_size)]

    # Process in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        plot_results = pool.map(plot_points, chunks)

    # Combine plots and show
    combined_fig, combined_ax = combine_plots(plot_results)
    plt.show()

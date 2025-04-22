import multiprocessing
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from matplotlib import MatplotlibDeprecationWarning
from matplotlib import cm
from matplotlib.colors import Normalize
from Graphs.SE_Asia_Earthquakes_processing import coords_try,magnitude_selector  # Ensure the import path is correct

# Suppress specific warnings, but be more specific about which warnings to ignore
warnings.filterwarnings("ignore", category=UserWarning, module='cartopy')
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning, module='mpld3')

# Function to plot data points with magnitude-based color
def plot_points(data_point):
    # Return the coordinates along with the magnitude for plotting
    return data_point

# Function to combine plots into a single figure
# Function to combine plots into a single figure
# Function to combine plots into a single figure
def combine_plots(plot_data):
    # Create figure and axis with projection (increase dpi for better quality)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=600)

    # Set the zoomed-in extent to focus on a specific point or region
    # Modify these values for different zoom levels
    # ax.set_extent([100.3, 100.7, 13.5, 14], crs=ccrs.PlateCarree())  # Example: these are the Bangkok Coordinates

    # Add map features (land, ocean, etc.)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle="--")
    ax.add_feature(cfeature.STATES, linestyle=":")

    # Extract magnitudes from the filtered data points
    magnitudes = [point[2] for sublist in plot_data for point in sublist]
    norm = Normalize(vmin=min(magnitudes), vmax=max(magnitudes))  # Normalize based on the min/max magnitude
    cmap = cm.viridis  # Choose a colormap

    # Plot points with color based on the magnitude
    for data_point in plot_data:
        for lat, lon, magnitude, time in data_point:  # Unpack all four values
            color = cmap(norm(magnitude))  # Map the magnitude to a color
            ax.plot(lon, lat, marker='o', color=color, markersize=5, transform=ccrs.PlateCarree())

    # Add a colorbar to indicate the magnitude scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for the colorbar to work
    fig.colorbar(sm, ax=ax, orientation="vertical", label="Magnitude")

    # Add latitude and longitude gridlines with labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # Disable labels on the top
    gl.right_labels = False  # Disable labels on the right
    gl.xlabel_style = {'size': 10, 'color': 'black'}  # Style for longitude labels
    gl.ylabel_style = {'size': 10, 'color': 'black'}  # Style for latitude labels

    ax.set_title(f"Earthquakes of Magnitude {magnitude_selector}+ in SE Asia 1999-2025", fontsize=14, pad=20)

    return fig, ax


# This block ensures the script runs correctly when using multiprocessing on Windows
if __name__ == '__main__':
    # Split the data into smaller chunks
    num_processes = multiprocessing.cpu_count()  # Get the number of available cores
    chunk_size = max(1, len(coords_try) // num_processes)
    # Divide the data for each core
    chunks = [coords_try[i:i + chunk_size] for i in range(0, len(coords_try), chunk_size)]

    # Use multiprocessing to process and get the coordinates in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        plot_results = pool.map(plot_points, chunks)

    # After multiprocessing, combine the individual plot results into one
    combined_fig, combined_ax = combine_plots(plot_results)
    plt.show(block=False)  # Non-blocking display
    # Show the combined map with all points plotted
    plt.show()

import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have Qt installed

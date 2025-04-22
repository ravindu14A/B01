import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from matplotlib.colors import Normalize
from Graphs.SE_Asia_Earthquakes_processing import coords_try, magnitude_selector

# Function to combine plots into a single figure
def combine_plots(plot_data):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=600)

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle="--")
    ax.add_feature(cfeature.STATES, linestyle=":")

    # Extract magnitudes and normalize
    magnitudes = [point[2] for point in plot_data]
    norm = Normalize(vmin=min(magnitudes), vmax=max(magnitudes))
    cmap = cm.viridis

    # Plot points with color based on magnitude
    for lat, lon, magnitude, time in plot_data:
        color = cmap(norm(magnitude))
        ax.plot(lon, lat, marker='o', color=color, markersize=5, transform=ccrs.PlateCarree())

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical", label="Magnitude")

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    ax.set_title(f"Earthquakes of Magnitude {magnitude_selector}+ in SE Asia 1999-2025", fontsize=14, pad=20)

    return fig, ax

# Main script
if __name__ == '__main__':
    # Combine the data into a single plot
    combined_fig, combined_ax = combine_plots(coords_try)
    plt.show()
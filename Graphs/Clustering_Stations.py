import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import great_circle
from sklearn.cluster import DBSCAN
from matplotlib.patches import Circle
import os
from datetime import datetime


def load_and_preprocess(filepath):
    """Load and preprocess station data"""
    df = pd.read_csv(filepath)
    df['Latitude_degrees'] = np.degrees(df['Average Latitude'])
    df['Longitude_degrees'] = np.degrees(df['Average Longitude'])
    return df


def calculate_distances(df):
    """Calculate pairwise distances between stations"""
    coords = df[['Latitude_degrees', 'Longitude_degrees']].values
    return [great_circle(coords[i], coords[j]).km
            for i in range(len(coords))
            for j in range(i + 1, len(coords))]


def perform_clustering(df, eps_km):
    """Perform DBSCAN clustering"""
    eps_rad = eps_km / 6371  # Earth's radius in km
    coords_rad = df[['Average Latitude', 'Average Longitude']].values
    clustering = DBSCAN(eps=eps_rad, min_samples=2, metric='haversine').fit(coords_rad)
    df['Cluster'] = clustering.labels_
    return df, clustering


def plot_clusters(df, eps_km, output_dir):
    """Plot clusters on map with circles showing cluster extent"""
    try:
        fig = plt.figure(figsize=(12, 8), dpi=1200)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Enhanced map features
        ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='black', facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', edgecolor='black', alpha=0.5)

        # Prepare colors using a perceptually uniform colormap
        unique_clusters = np.unique(df['Cluster'][df['Cluster'] != -1])
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

        # Plot each cluster with enhanced styling
        for i, cluster in enumerate(unique_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            color = colors[i]

            # Plot stations with better markers
            ax.scatter(cluster_data['Longitude_degrees'], cluster_data['Latitude_degrees'],
                       color=color, marker='o', s=40, edgecolor='white', linewidth=0.5,
                       label=f'Cluster {cluster}', zorder=5, transform=ccrs.PlateCarree())

            # Calculate cluster center and radius with enhanced precision
            center_lon = np.median(cluster_data['Longitude_degrees'])
            center_lat = np.median(cluster_data['Latitude_degrees'])
            radius = max(great_circle((center_lat, center_lon), (lat, lon)).km
                         for lat, lon in zip(cluster_data['Latitude_degrees'],
                                             cluster_data['Longitude_degrees']))

            # Add circle with better styling
            circle = Circle((center_lon, center_lat), radius / 111,
                            edgecolor=color, facecolor=color, alpha=0.15,
                            linewidth=1.5, linestyle='-', transform=ccrs.PlateCarree())
            ax.add_patch(circle)

        # Plot single stations (formerly "noise") with improved styling
        single_stations = df[df['Cluster'] == -1]
        if not single_stations.empty:
            ax.scatter(single_stations['Longitude_degrees'], single_stations['Latitude_degrees'],
                       color='#7f7f7f', marker='x', s=60, linewidth=1.5,
                       label='No cluster', zorder=5, transform=ccrs.PlateCarree())

        # Set map extent with dynamic buffer
        data_ratio = (df['Longitude_degrees'].max() - df['Longitude_degrees'].min()) / \
                     (df['Latitude_degrees'].max() - df['Latitude_degrees'].min())
        buffer_factor = min(1.5, 0.5 * data_ratio)  # Prevent excessive buffering
        lon_buffer = buffer_factor * eps_km / 111
        lat_buffer = buffer_factor * eps_km / 111

        ax.set_extent([
            df['Longitude_degrees'].min() - lon_buffer,
            df['Longitude_degrees'].max() + lon_buffer,
            df['Latitude_degrees'].min() - lat_buffer,
            df['Latitude_degrees'].max() + lat_buffer
        ], crs=ccrs.PlateCarree())

        # Enhanced grid and title
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}

        ax.set_title(f"Ground Station Clustering (ε={eps_km} km)", fontsize=12, pad=15, weight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)

        plt.tight_layout()

        # Save the figure with high quality settings (removed quality parameter for PNG)
        output_path = os.path.join(output_dir, f'clusters_eps_{eps_km}km.png')
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        print(f"Saved high-quality plot for ε={eps_km} km to {output_path}")

        # Display the plot
        plt.show()

    except Exception as e:
        print(f"Error generating plot for ε={eps_km} km: {str(e)}")
    finally:
        plt.close(fig)


def main():
    # Create output directory with timestamp
    output_dir = f"cluster_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    filepath = r'C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\SE_Asia\SE_Asia_stations_with_20plus.csv'
    df = load_and_preprocess(filepath)

    # Calculate distances for reference
    distances = calculate_distances(df)
    print("\nDistance Statistics:")
    print(f"Min distance: {min(distances):.2f} km")
    print(f"Max distance: {max(distances):.2f} km")
    print(f"Median distance: {np.median(distances):.2f} km")
    print(f"Suggested EPS range: {np.median(distances) * 0.2:.1f}-{np.median(distances) * 0.6:.1f} km")

    # Loop through EPS values from 20 to 100 km in steps of 10 km
    for eps_km in range(20, 101, 10):
        print(f"\nProcessing ε={eps_km} km")

        # Perform clustering
        clustered_df, clustering = perform_clustering(df.copy(), eps_km)

        # Print cluster info
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_single = sum(clustering.labels_ == -1)
        print(f"Found {n_clusters} clusters and {n_single} single stations")

        # Generate, save, and display plot
        plot_clusters(clustered_df, eps_km, output_dir)

        # Save clustered data
        output_csv = os.path.join(output_dir, f'clusters_eps_{eps_km}km.csv')
        clustered_df.to_csv(output_csv, index=False)
        print(f"Saved cluster data to {output_csv}")

    print("\nAll EPS iterations completed! Results saved to:", os.path.abspath(output_dir))


if __name__ == '__main__':
    main()
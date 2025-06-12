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


def load_data(path):
    df = pd.read_csv(path)
    df['lat_deg'] = np.degrees(df['Average Latitude'])
    df['lon_deg'] = np.degrees(df['Average Longitude'])
    return df


def pairwise_distances(df):
    coords = df[['lat_deg', 'lon_deg']].values
    return [great_circle(coords[i], coords[j]).km for i in range(len(coords)) for j in range(i + 1, len(coords))]


def deduplicate_clusters(df):
    out = {}
    for _, row in df[df['Cluster'] != -1].iterrows():
        out.setdefault(row['Cluster'], set()).add(row['File'])
    return pd.DataFrame([{'Cluster': c, 'NumStations': len(s), 'StationFile': list(s)} for c, s in out.items()])


def cluster_stations(df, eps_km):
    eps_rad = eps_km / 6371
    unique = df.drop_duplicates('File')
    coords = unique[['Average Latitude', 'Average Longitude']].values
    labels = DBSCAN(eps=eps_rad, min_samples=2, metric='haversine').fit(coords).labels_
    unique['Cluster'] = labels
    merged = df[['File', 'lat_deg', 'lon_deg']].merge(unique[['File', 'Cluster']], on='File', how='left').fillna(-1)
    return merged, labels


def draw_clusters(df, eps_km, out_dir):
    fig = plt.figure(figsize=(12, 8), dpi=1200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', edgecolor='black', alpha=0.5)

    clusters = np.unique(df['Cluster'][df['Cluster'] != -1])
    colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))

    for i, c in enumerate(clusters):
        d = df[df['Cluster'] == c]
        ax.scatter(d['lon_deg'], d['lat_deg'], color=colors[i], edgecolor='white', s=40, linewidth=0.5, zorder=5)

        lat_c, lon_c = np.median(d['lat_deg']), np.median(d['lon_deg'])
        r_km = max(great_circle((lat_c, lon_c), (lat, lon)).km for lat, lon in zip(d['lat_deg'], d['lon_deg']))
        circ = Circle((lon_c, lat_c), r_km / 111, color=colors[i], alpha=0.15, transform=ccrs.PlateCarree())
        ax.add_patch(circ)

    noise = df[df['Cluster'] == -1]
    ax.scatter(noise['lon_deg'], noise['lat_deg'], color='gray', marker='x', s=60, linewidth=1.5, zorder=5)

    lon_buf = eps_km / 111
    lat_buf = eps_km / 111
    ax.set_extent([
        df['lon_deg'].min() - lon_buf,
        df['lon_deg'].max() + lon_buf,
        df['lat_deg'].min() - lat_buf,
        df['lat_deg'].max() + lat_buf
    ])

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"Station Clustering (eps={eps_km} km)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'clusters_eps_{eps_km}km.png')
    plt.savefig(out_path, dpi=1200)
    print("Saved:", out_path)
    plt.close(fig)


def main():
    out_dir = f"cluster_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    path = r'C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\SE_Asia\SE_Asia_stations_with_20plus.csv'
    df = load_data(path)

    dists = pairwise_distances(df)
    print(f"Distance range: {min(dists):.1f}-{max(dists):.1f} km | Median: {np.median(dists):.1f} km")

    for eps in range(100, 121, 10):
        clustered, labels = cluster_stations(df, eps)
        clean = deduplicate_clusters(clustered)

        print(f"\nε={eps} km → {len(clean)} clusters, {sum(clustered['Cluster'] == -1)} single stations")
        for _, r in clean.iterrows():
            print(f"Cluster {r['Cluster']}: {r['NumStations']} → {r['StationFile']}")

        clustered.to_csv(os.path.join(out_dir, f'clusters_eps_{eps}km.csv'), index=False)
        clean.to_csv(os.path.join(out_dir, f'clean_cluster_summary_eps_{eps}km.csv'), index=False)
        draw_clusters(clustered, eps, out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()

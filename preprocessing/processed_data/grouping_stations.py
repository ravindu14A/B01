import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle

# Read data (coordinates are in radians)
station_coord_raw = pd.read_csv(r'C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\SE_Asia\SE_Asia_stations_with_20plus.csv')

# from radians to degrees
station_coord_raw['Latitude_degrees'] = np.degrees(station_coord_raw['Average Latitude'])
station_coord_raw['Longitude_degrees'] = np.degrees(station_coord_raw['Average Longitude'])

# Create coordinate arrays in both radians (for DBSCAN) and degrees (for great_circle)
station_coords_degrees = np.column_stack((station_coord_raw['Latitude_degrees'],
                                         station_coord_raw['Longitude_degrees']))
station_coords_radians = np.column_stack((station_coord_raw['Average Latitude'],
                                         station_coord_raw['Average Longitude']))

# Calculate pairwise distances in kilometers using degrees
distances = []
for i in range(len(station_coords_degrees)):
    for j in range(i + 1, len(station_coords_degrees)):
        distances.append(great_circle(station_coords_degrees[i], station_coords_degrees[j]).km)

print("\nDistance Statistics:")
print(f"Min distance: {min(distances):.2f} km")
print(f"Max distance: {max(distances):.2f} km")
print(f"Median distance: {np.median(distances):.2f} km")

# Calculate suggested EPS
suggested_eps = np.median(distances) * 0.4
print(f"\nSuggested EPS (0.4 × median distance): {suggested_eps:.2f} km")

# Get user input for EPS
while True:
    try:
        user_eps = float(input(f"Enter EPS value in km (suggested: {suggested_eps:.2f}): "))
        if user_eps <= 0:
            print("EPS must be positive. Try again.")
            continue
        break
    except ValueError:
        print("Please enter a valid number.")

# Apply DBSCAN clustering (using radians)
clustering = DBSCAN(eps=user_eps/6371, min_samples=2, metric='haversine').fit(station_coords_radians)
station_coord_raw['Cluster'] = clustering.labels_

# Print cluster results
print("\nCluster Results:")
unique_clusters = np.unique(clustering.labels_)
for cluster in unique_clusters:
    if cluster != -1:
        print(f"Cluster {cluster} (Size: {sum(clustering.labels_ == cluster)}):")
        print(station_coord_raw[station_coord_raw['Cluster'] == cluster]['File'].tolist())
    else:
        print(f"Noise points (-1) (Count: {sum(clustering.labels_ == -1)}):")
        print(station_coord_raw[station_coord_raw['Cluster'] == -1]['File'].tolist())

# Save results (including both coordinate formats)
station_coord_raw.to_csv('grouped_stations.csv', index=False)
print("\nClustering complete. Results saved to 'grouped_stations.csv'")
print("\nFinal cluster assignments:")
print(station_coord_raw[['File', 'Cluster', 'Latitude_degrees', 'Longitude_degrees']])



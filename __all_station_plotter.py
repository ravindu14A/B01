import DataFrame_genrator
df = DataFrame_genrator.GetSavedData()


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

# Ensure Date is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Get the first entry per station
df_first = df.sort_values("Date").groupby("Station").first().reset_index()

# Ensure Latitude & Longitude are numeric
df_first["Latitude"] = pd.to_numeric(df_first["Latitude"], errors="coerce")
df_first["Longitude"] = pd.to_numeric(df_first["Longitude"], errors="coerce")

# Convert radians to degrees
df_first["Latitude"] = np.degrees(df_first["Latitude"])
df_first["Longitude"] = np.degrees(df_first["Longitude"])

# Print min/max values for debugging
print("Latitude range (degrees):", df_first["Latitude"].min(), "to", df_first["Latitude"].max())
print("Longitude range (degrees):", df_first["Longitude"].min(), "to", df_first["Longitude"].max())

# Remove invalid latitude/longitude values
df_first = df_first[(df_first["Latitude"].between(-90, 90)) & (df_first["Longitude"].between(-180, 180))]

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df_first, geometry=gpd.points_from_xy(df_first["Longitude"], df_first["Latitude"]))

# Set CRS to WGS84
gdf.crs = "EPSG:4326"

# Plot using geopandas with contextily base map
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the stations as points
gdf.plot(ax=ax, color='purple', markersize=50, label="Stations")

# Fix aspect ratio issue
ax.set_aspect("auto")

# Add the basemap (OpenStreetMap)
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Annotate station names next to the points
for i, row in gdf.iterrows():
    ax.text(row.geometry.x + 0.02, row.geometry.y + 0.02, row["Station"], fontsize=9)

# Set title and labels
ax.set_title("Positions of First Entry for Each Station")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Display the plot
plt.tight_layout()
plt.show()

import DataFrame_genrator
df = DataFrame_genrator.GetSavedData()


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

# Ensure Date is in datetime format
df["date"] = pd.to_datetime(df["date"])

# Get the first entry per station
df_first = df.sort_values("date").groupby("station").first().reset_index()

# Ensure Latitude & Longitude are numeric
df_first["lat"] = pd.to_numeric(df_first["lat"], errors="coerce")
df_first["long"] = pd.to_numeric(df_first["long"], errors="coerce")

# Convert radians to degrees
df_first["lat"] = np.degrees(df_first["lat"])
df_first["long"] = np.degrees(df_first["long"])

# Print min/max values for debugging
print("Latitude range (degrees):", df_first["lat"].min(), "to", df_first["lat"].max())
print("Longitude range (degrees):", df_first["long"].min(), "to", df_first["long"].max())

# Remove invalid latitude/longitude values
df_first = df_first[(df_first["lat"].between(-90, 90)) & (df_first["long"].between(-180, 180))]

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df_first, geometry=gpd.points_from_xy(df_first["long"], df_first["lat"]))

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
    ax.text(row.geometry.x + 0.02, row.geometry.y + 0.02, row["station"], fontsize=9)

# Set title and labels
ax.set_title("Positions of First Entry for Each Station")
ax.set_xlabel("long")
ax.set_ylabel("lat")

# Display the plot
plt.tight_layout()
plt.show()

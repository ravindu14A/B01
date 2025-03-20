import main

df = main.GetSavedData()


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

# Ensure Date is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Prompt for a specific date
specified_date = input("Enter the date (YYYY-MM-DD): ")

# Convert the specified date to datetime
specified_date = pd.to_datetime(specified_date)

# Filter data for the specified date
df_on_date = df[df["Date"] == specified_date]

# Convert radians to degrees
df_on_date["Longitude"] = np.degrees(df_on_date["Longitude"])  # Convert longitude from radians to degrees
df_on_date["Latitude"] = np.degrees(df_on_date["Latitude"])    # Convert latitude from radians to degrees

# Convert the dataframe to a GeoDataFrame
gdf = gpd.GeoDataFrame(df_on_date, geometry=gpd.points_from_xy(df_on_date["Longitude"], df_on_date["Latitude"]))

# Set the coordinate reference system (CRS) to WGS84 (latitude/longitude)
gdf.crs = "EPSG:4326"

# Plot using geopandas with contextily base map
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the stations as points
gdf.plot(ax=ax, color='purple', markersize=50, label="Stations")

# Add the basemap (OpenStreetMap)
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Annotate station names next to the points
for i, row in gdf.iterrows():
    ax.text(row.geometry.x + 0.02, row.geometry.y + 0.02, row["Station"], fontsize=9)

# Set title and labels
ax.set_title(f"Positions of All Stations on {specified_date.date()}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Display the plot
plt.tight_layout()
plt.show()

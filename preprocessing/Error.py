from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

def add_point(lat, long):
    ax.plot(long, lat, marker='o', color='blue', markersize=1)

def add_arrow(lat1, long1, lat2, long2):
    ax.annotate("",
                xy=(lon2, lat2), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                xytext=(lon1, lat1), textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle="->", color='red', lw=2))


directory = Path("processed_data/Malaysia")

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.set_extent([np.degrees(1.7718096161158234)-0.00001, np.degrees(1.7718096161158234)+0.00001, np.degrees(0.06571810956388711)-0.00001, np.degrees(0.06571810956388711)+0.00001])
# for file in directory.iterdir():
with open(f'processed_data/Malaysia/BEHR.pkl', 'rb') as f:
    data = pickle.load(f)
    lat = data["lat"]
    long = data["long"]
    for lat, long in zip(lat, long):

        add_point(np.degrees(lat),np.degrees(long))

plt.show()


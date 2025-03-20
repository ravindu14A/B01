import zipfile
import os
import cartopy.io.shapereader as shapereader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# Path to your zip file
zip_file_path = r"C:\Users\nicov\Downloads\th_shp.zip"
extract_folder = r"C:\Users\nicov\Downloads\th_shp"  # Folder to extract the files to

# Extract files from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Path to the extracted shapefile (ensure the filename matches exactly)
shapefile_path = r"C:\Users\nicov\Downloads\th_shp\th.shp"  # Ensure this points to the correct .shp file

# Read the shapefile using Cartopy's shapereader
shapefile = shapereader.Reader(shapefile_path)

# Create a map using PlateCarree projection
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Set extent to zoom in on Bangkok (adjust the bounding box)
ax.set_extent([100, 101, 13, 14.5], crs=ccrs.PlateCarree())  # Zoom in around Bangkok

# Add map features like land, ocean, coastline, and borders
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle="--")

# Plot the shapefile geometries on the map
ax.add_geometries(shapefile.geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='red', linewidth=1)

# Add a title to the map
ax.set_title("Shapefile Data Zoomed into Bangkok", fontsize=14)

from PIL import Image

# Open the TIFF file
img = Image.open("C:\Users\nicov\Downloads\gebco_2023_sub_ice_n90.0_s0.0_w-90.0_e0.0.tif")

# Show the image
img.show()

# To access pixel data (as an example):
pixels = img.load()
print(pixels[0, 0])  # Print the pixel value at position (0, 0)

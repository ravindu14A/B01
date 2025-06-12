import pickle
import pandas as pd
import numpy as np
import os
from pyproj import Geod


# Load data
country = "malaysia"

geod = Geod(ellps="WGS84")

def get_displacement_mm(lat0, lon0, lat, lon):
    az12, az21, dist_m = geod.inv(np.degrees(lon0), np.degrees(lat0), np.degrees(lon), np.degrees(lat))
    az_rad = np.radians(az12)
    d_north = dist_m * np.cos(az_rad) *100 # cm
    d_east = dist_m * np.sin(az_rad) *100 # cm
    return d_north, d_east

directory = f"../data/partially_processed_steps/{country}/filtered"

directory_out = f"../data/partially_processed_steps/{country}/filtered_cm"

for filename in os.listdir(directory_out):
    file_path = os.path.join(directory_out, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for filename in os.listdir(directory):
    north = [0]
    east = [0]
    counter = 0
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory, filename)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)

            for lat,long in zip(data["lat"][1:], data["long"][1:]):
                d_north, d_east = get_displacement_mm(data["lat"][counter], data["long"][counter], lat, long)
                north.append(d_north)
                east.append(d_east)

            counter += 1

    north_array = np.array(north)
    east_array = np.array(east)

    df["lat"] = north_array
    df["long"] = east_array

    filename = os.path.join(directory_out, f"{filename}")
    df.to_pickle(filename)









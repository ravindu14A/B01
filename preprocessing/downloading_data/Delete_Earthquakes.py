import numpy as np
import pandas as pd
import os
import pickle

country = "Malaysia"
directory = f"../processed_data/{country}/Filtered_cm_normalised"

directory_out = f"../processed_data/{country}/Filtered_cm"

for filename in os.listdir(directory_out):
    file_path = os.path.join(directory_out, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory, filename)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)

            # First differences
            diff_north = df['lat'].diff()
            diff_east = df['long'].diff()

            # Choose a threshold (e.g. >2 cm/day is suspicious)
            threshold_cm = 2.0
            jump_north = diff_north.abs() > threshold_cm
            jump_east = diff_east.abs() > threshold_cm

            # Combine both directions
            jumps = jump_north | jump_east
            jump_times = gps_df.index[jumps]
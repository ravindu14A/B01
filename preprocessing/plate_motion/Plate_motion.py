import os
import pickle
import pandas as pd
from datetime import timedelta

"""
Deleting the relative motion of the plate
"""

country = "Thailand"
North = -1.15821  #cm/year
East = 2.9768 #cm/year

North = North/365.25
East = East/365.25


directory = f"../processed_data/{country}/Filtered_cm"

directory_out = f"../processed_data/{country}/Filtered_cm_normalised"

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

            begin = df["date"].iloc[0]
            days = (df["date"] - begin) / pd.Timedelta(days=1)

            df["lat"] = df["lat"] - North * days
            df["long"] = df["long"] - East * days

    filename = os.path.join(directory_out, f"{filename}")
    df.to_pickle(filename)
import os
import FileReader
import numpy as np
import pandas as pd
folder_path = 'data'

# List all files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

df = pd.DataFrame(columns=["Date", "Station", "Position", "Covariance"])

for file in files:
	coordinates, covariance_matrices, date_obj = FileReader.getDayData(r"data\\"+file)
	with open('stations.txt', 'r') as file:
		station_names = [line.strip() for line in file.readlines()]
		stations_names = [station.strip() for station in station_names]

	mask =~(coordinates == 0).all(axis=1)

	filtered_coordinates = coordinates[mask]
	filtered_covariance_matrices = covariance_matrices[mask]
	date_list = [date_obj] * len(filtered_coordinates)
	stations_names = np.array(stations_names)
	filtered_station_names = stations_names[mask]

	new_data = pd.DataFrame({
	"Date": date_list,
	"Station": filtered_station_names,
	"Position": list(filtered_coordinates),
	"Covariance": list(filtered_covariance_matrices)
	})
	df = pd.concat([df, new_data], ignore_index=True)

df.sort_index
df.to_csv("output1.csv",index=False)
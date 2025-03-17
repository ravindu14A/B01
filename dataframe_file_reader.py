import pandas as pd
import numpy as np
import FileReader

df = pd.DataFrame(columns=["Date", "Station", "Position", "Covariance"])

coordinates, covariance_matrices, date_obj = FileReader.getDayData("PFITRF14284.21C")

with open('stations.txt', 'r') as file:
	station_names = [line.strip() for line in file.readlines()]
	stations_names = [station.strip() for station in station_names]

mask =~(coordinates == 0).all(axis=1)

filtered_coordinates = coordinates[mask]
filtered_covariance_matrices = covariance_matrices[mask]
date_list = [date_obj] * len(filtered_coordinates)
stations_names = np.array(stations_names)
# print(stations_names[:3])
# print(type(mask))  # Should be a NumPy array or Pandas Series
# print(mask.dtype)  # Should be 'bool'
# print(mask.shape, stations_names.shape)  # Should be same shape
filtered_station_names = stations_names[mask]

print("-----------------------------")
print(filtered_coordinates[:4])
print(filtered_covariance_matrices[:4])
print(date_list[:4])
print(filtered_station_names[:4])

new_data = pd.DataFrame({
    "Date": date_list,
    "Station": filtered_station_names,
    "Position": list(filtered_coordinates),
    "Covariance": list(filtered_covariance_matrices)
})
df = pd.concat([df, new_data], ignore_index=True)
print(df)
#df.to_csv("output.csv", index=False)

print(df.loc[df["Station"] == "ARAU", "Covariance"])
a = np.array(df.loc[df["Station"] == "ARAU", "Covariance"].iloc[0])
print(a)
print(a[1,2])
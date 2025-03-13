import numpy as np
from datetime import datetime

with open('stations.txt', 'r') as file:
	station_names = [line.strip() for line in file.readlines()]
	stations_names = [station.strip() for station in station_names]


def getDayData(filename):
	with open(filename, "r") as file:
		first_line = file.readline()
		first_line = first_line.split()

		N_parameters = int(first_line[0])
		N_stations= int(N_parameters/3)

		date_str = first_line[3]
		date_str = date_str.replace('.', '').strip()  # Remove any periods or extra spaces
		date_obj = datetime.strptime(date_str, r'%y%b%d')
		date_timestamp = date_obj.timestamp()

		coordinates = np.zeros((len(station_names),3))
		covariance_matrices = np.zeros((len(station_names),3,3))

		for i in range(N_stations):
			for j in range(3):
				line = file.readline()
				line = line.split()
				name = line[1]
				#important bit 1
				name_index = station_names.index(name) # get the index of the station name(not sure they are alsways order, some are missing)
				coordinates[name_index,j] = float(line[4])
				std = float(line[6])
				covariance_matrices[name_index,j,j] = std**2
				####

		for i in range(N_stations):
			for j in range(3):
				line = file.readline()
				line=line.split()
				index_1 = int(line[0])%3 - 1
				index_2 = int(line[1])%3 - 1

				name_index = station_names.index(line[3])

				#important bit 2
				correlation = float(line[2])
				covariance = correlation * np.sqrt(covariance_matrices[name_index,index_1,index_1]) * np.sqrt(covariance_matrices[name_index,index_2,index_2])
				covariance_matrices[name_index,index_1,index_2] = covariance
				covariance_matrices[name_index,index_2,index_1] = covariance
				#---


		return coordinates, covariance_matrices, date_obj
		
	
coord, cov, dat = getDayData('PFITRF14284.21C')
print(cov[:4])


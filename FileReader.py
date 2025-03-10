import numpy as np

#day = 366 for current thingy
def getDayData(day):

	#PZTIRF is some convention
	# 08 is the reference frame IGS08
	# other part is day
	with open("PZITRF08" + str(day) + ".12X", "r") as file:
		first_line = file.readline()
		first_line = first_line.split()
		N_parameters = int(first_line[0])
		N_stations= int(N_parameters/3)

		names = ['']*N_stations
		coordinates = np.zeros((N_stations,3))
		covariance_matrices = np.zeros((N_stations,3,3))

		for i in range(N_stations):
			for j in range(3):
				line = file.readline()
				line = line.split()
				coordinates[i,j] = float(line[4])
				covariance_matrices[i,j,j] = float(line[6])
			names[i] = line[1]

		for i in range(N_stations):
			for j in range(3):
				line = file.readline()
				line=line.split()
				index_1 = int(line[0])%3 - 1
				index_2 = int(line[1])%3 - 1
				covariance_matrices[i,index_1,index_2] = float(line[2])
				covariance_matrices[i,index_2,index_1] = float(line[2])

	return names, coordinates, covariance_matrices
		# print(names[:5])
		# print(coordinates[:5])
		# print(covariance_matrices[:5])
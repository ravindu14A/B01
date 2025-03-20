import numpy as np
import pandas as pd
from datetime import datetime

def GetDataFrame(file_path):
	with open(file_path, 'r') as f:
		first_line = f.readline()
		first_line = first_line.split()
		# Extract date from first line
		date_str = first_line[3]
		date_str = date_str.replace('.', '').strip()  # Remove any periods or extra spaces
		date_obj = datetime.strptime(date_str, r'%y%b%d')
		#date_timestamp = date_obj.timestamp()
		
		N_parametsrs = int(first_line[0])

		N_stations = int(N_parametsrs/3)

		station_names = []
		positions = np.zeros((N_stations,3))
		stds = np.zeros((N_stations,3))
		cov_matrices = np.zeros((N_stations,3,3))

		for i in range(N_stations):
			for j in range(3):
				line = f.readline()
				line = line.split()

				positions[i,j] = float(line[4])

				stds[i,j] = float(line[6])
				cov_matrices[i,j,j] = float(line[6])**2

				station_name = line[1]
				if station_name not in station_names:
					station_names.append(station_name)


		while True:#for i in range(int((N_stations*3-2)*(N_stations*3-1)/2)):
			
			line = f.readline()
			line = line.split()
			try: # this alongside the while True is very illegal but fuck it
				index1 = int(line[0])-1
				index2 = int(line[1])-1
			except:
				break
			
			if index1//3 == index2//3:
				station_index = index1//3
				
				cov_matrices[station_index,index1%3,index2%3] = float(line[2]) *  stds[station_index,index1%3] * stds[station_index,index2%3]
				cov_matrices[station_index,index2%3,index1%3] = cov_matrices[station_index,index1%3,index2%3]

				#if cov_matrices[station_index,index1%3,index2%3]==0:
				#		raise Exception("wtf???")

			#if cov_matrices[1,0,0] != 1.5384770e-06:
		#		raise Exception("how")
				
		dates = [date_obj]*N_stations
		df=pd.DataFrame(zip(dates,station_names,positions,cov_matrices),columns=["Date", "Station", "Position", "Covariance"])
		return df
	

# df = GetDataFrame(r"data\\PFITRF14003.05C")


# print(df)
#df = GetDataFrame(r'data\PFITRF14003.08C')
#print(df["Covariance"].iloc[-1])
# import Coordinate
# df["Converted Covariance"] = df.apply(lambda row: Coordinate.convert2electricboogaloo(row["Position"], row["Covariance"]), axis=1)


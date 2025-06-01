import pandas as pd
import numpy as np
import os
from datetime import datetime

# input file path to PFITRF
# returns df format ["date", "station", "xyz_position", "xyz_covariance"]
def readFile(file_path):
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
		df=pd.DataFrame(zip(dates,station_names,positions,cov_matrices),columns=["date", "station", "xyz_position", "xyz_covariance"])
		return df
	
def getFilePaths(directory):
	folder_path = directory
	#pattern = re.compile(r'^PFITRF14.*\..*C$')
	file_paths = []
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			file_paths.append(os.path.join(root, file))
	return file_paths


#read all files return df "date", "station", "xyz_position", "xyz_covariance" 
#ordered by date
def readAllFiles():

	#data_directory = 'data'
	filepaths_thailand = getFilePaths(r"raw_data\thailand")
	filepaths_malaysia = getFilePaths(r"raw_data\malaysia")

	all_filepaths = filepaths_malaysia + filepaths_thailand

	# generate DF with x,y,z as vector
	df = pd.DataFrame(columns=["date", "station", "xyz_position", "xyz_covariance"])
	for filepath in all_filepaths:
		newdf = readFile(filepath)
		if df.empty:
			df = newdf
		df = pd.concat([df, newdf], ignore_index=True)
	
	#split into x,y,z seperate
	df[['x', 'y', 'z']] = pd.DataFrame(df['xyz_position'].tolist(), index=df.index)
	# Drop the 'pos' column
	df = df.drop(columns=['xyz_position'])
	df = df.sort_values(by = 'date')
	return df

def removeDuplicates(df):
	#print(df[(df['date'] == pd.Timestamp('2000-01-04')) & (df['station'] == 'NTUS')])

	df.drop_duplicates(subset=['date', 'station'], keep='first',inplace=True)
	#print(df[(df['date'] == pd.Timestamp('2000-01-04')) & (df['station'] == 'NTUS')])
	return df

from utils.geo_utils import xyz_to_geodetic, displacement

def addLatLonHeight(df):
	new_cols = df.apply(lambda row: xyz_to_geodetic(row['x'], row['y'], row['z']), axis=1)
	df[['lat', 'lon', 'h']] = pd.DataFrame(new_cols.tolist(), index=df.index)

def addRelativeDisplacementmm(df):
	for station, group in df.groupby("station"):

		lat1, lon1, h1 = group.iloc[0][["lat", "lon", "h"]]
		xyz1 = list(group.iloc[0][['x','y','z']])

		d_north, d_east = zip(*group.apply(lambda row: displacement(lat1, lon1, row["lat"], row["lon"], xyz1, [row['x'],row['y'],row['z']]), axis=1))
		d_up = (group["h"] - h1) * 1000  # in mm

		df.loc[group.index, "d_north_mm"] = d_north
		df.loc[group.index, "d_east_mm"] = d_east
		df.loc[group.index, "d_up_mm"] = d_up
	return df

def convert_xyz_cov_to_enu(df):
	"""
	Convert the 'xyz_covariance' (3x3 matrix in m²) in ECEF coordinates 
	to ENU covariance (in cm²) for each row using corresponding 'lat' and 'lon'.

	Adds a new column 'enu_covariance_cm2' with the resulting 3x3 matrix.
	This function modifies the DataFrame in-place and also returns it.
	"""
	def rotation_matrix_ecef_to_enu(lat_deg, lon_deg):
		# Convert lat/lon from degrees to radians
		lat = np.radians(lat_deg)
		lon = np.radians(lon_deg)

		# Rotation matrix from ECEF to ENU
		R = np.array([
			[-np.sin(lon),              np.cos(lon),              0],
			[-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
			[ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
		])
		return R

	def transform_covariance(row):
		cov_xyz = np.array(row['xyz_covariance'])
		R = rotation_matrix_ecef_to_enu(row['lat'], row['lon'])

		# Rotate covariance to ENU frame
		cov_enu = R @ cov_xyz @ R.T

		# Convert from m² to cm²
		cov_enu_cm2 = cov_enu * 1e4

		return cov_enu_cm2

	# Apply to each row and store new column
	df['enu_covariance_cm2'] = df.apply(transform_covariance, axis=1)

	return df



def generatePreprocessedDF():
	df = readAllFiles()
	removeDuplicates(df)
	addLatLonHeight(df)
	addRelativeDisplacementmm(df)
	convert_xyz_cov_to_enu(df)
	return df
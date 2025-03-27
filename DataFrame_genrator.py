import os
import re
import pandas as pd
from FileReader import GetDataFrame
from Coordinate import getGeodetic
import numpy as np

def get_filenames(directory):
	pattern = re.compile(r'^PFITRF14.*\..*C$')
	filenames = [f for f in os.listdir(directory) if pattern.match(f)]
	return filenames

def generate_Dataframe_vector_xyz(data_directory):

	data_directory = 'data'
	filenames = get_filenames(data_directory)

	df = pd.DataFrame(columns=["date", "station", "Position", "covariance xyz"])
	for i in filenames:
		newdf = GetDataFrame(data_directory+ r'\\' + i)
		df = pd.concat([df, newdf], ignore_index=True)
	return df

def convert_to_columns_xyz(df):
	df_expanded = df['Position'].apply(pd.Series)
	df_expanded.columns = ['X', 'Y', 'Z']
	df = df.join(df_expanded)
	return df


def convert_to_degrees(df):
	df["Latitude deg"] = df["lat"].apply(np.degrees)
	df["Longitude deg"] = df["long"].apply(np.degrees)
	return df

def convert_to_mm_NE(df):
    # Constants (WGS84 ellipsoid)
    a = 6378.1370  # Semi-major axis [km]
    b = 6356.752314245  # Semi-minor axis [km]
    e2 = 1 - (b**2 / a**2)  # Eccentricity squared
    
    def geodetic_to_curvature_radius(latitude):
        """Calculate the radius of curvature in the meridian at a given latitude"""
        lat_rad = np.radians(latitude)
        R = a * (1 - e2) / (1 - e2 * np.sin(lat_rad)**2)**(3/2)
        return R

    def calculate_displacement(lat1, lon1, lat2, lon2):
        """Calculate the northward and eastward displacement in meters."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Radius of curvature at the first point (latitude)
        R = geodetic_to_curvature_radius(lat1)
        
        # Northward displacement (meters)
        delta_lat = lat2 - lat1
        delta_y = R * delta_lat  # Northward displacement
        
        # Eastward displacement (meters)
        delta_lon = lon2 - lon1
        delta_x = R * np.cos(lat1) * delta_lon  # Eastward displacement
        
        return delta_y * 1000, delta_x * 1000  # Convert to millimeters

    # Group by station and apply calculation for each station
    df['Distance_North_mm'] = np.nan
    df['Distance_East_mm'] = np.nan

    for station, group in df.groupby("station"):
        # Get the first entry for the station as the reference point
        ref_entry = group.iloc[0]
        lat1, lon1 = ref_entry["lat"], ref_entry["long"]
        
        # Iterate through the rest of the entries for the station
        for idx, row in group.iterrows():
            lat2, lon2 = row["lat"], row["long"]
            # Calculate displacement relative to the first entry
            delta_y_mm, delta_x_mm = calculate_displacement(lat1, lon1, lat2, lon2)
            
            # Update the displacement columns
            df.at[idx, 'Distance_North_mm'] = delta_y_mm
            df.at[idx, 'Distance_East_mm'] = delta_x_mm

    return df	

def save_pickle_Dataframe(df, name):
	df.to_pickle(r'output\\'+name)
	

def save_csv_Dataframe(df, name):
	df.to_csv(r'output\\'+name)


def convert_geodetic(df):
	df[["lat", "long", "Height"]] = df.apply(
    lambda row: pd.Series(getGeodetic(row["X"], row["Y"], row["Z"], row["covariance xyz"])[0]), axis=1
)

	df["covariance"] = df.apply(
		lambda row: getGeodetic(row["X"], row["Y"], row["Z"], row["covariance xyz"])[1], axis=1
	)
	return df
	


def SavePickleData():
    df = generate_Dataframe_vector_xyz('data')

    #filtered_sorted_df = df[df["station"] == "BABH"].sort_values(by="date")



    #print(filtered_sorted_df)
    df = convert_to_columns_xyz(df)

    df = convert_geodetic(df)

    save_pickle_Dataframe(df, r'pickle_geodetic_data_columns_xyz.pkl')


def GetSavedData():
    df = pd.read_pickle(r'output\\pickle_geodetic_data_columns_xyz.pkl')

    #print(df)
    return df

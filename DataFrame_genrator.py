import os
import re
import pandas as pd
from FileReader import GetDataFrame
from Coordinate import getGeodetic

def get_filenames(directory):
	pattern = re.compile(r'^PFITRF14.*\..*C$')
	filenames = [f for f in os.listdir(directory) if pattern.match(f)]
	return filenames

def generate_Dataframe_vector_xyz(data_directory):

	data_directory = 'data'
	filenames = get_filenames(data_directory)

	df = pd.DataFrame(columns=["Date", "Station", "Position", "Covariance"])
	for i in filenames:
		newdf = GetDataFrame(data_directory+ r'\\' + i)
		df = pd.concat([df, newdf], ignore_index=True)
	return df

def convert_to_columns_xyz(df):
	df_expanded = df['Position'].apply(pd.Series)
	df_expanded.columns = ['X', 'Y', 'Z']
	df = df.drop(columns=['Position']).join(df_expanded)
	return df

	


def save_pickle_Dataframe(df, name):
	df.to_pickle(r'output\\'+name)
	

def save_csv_Dataframe(df, name):
	df.to_csv(r'output\\'+name)


def convert_geodetic(df):
	df[["Latitude", "Longitude", "Height"]] = df.apply(
    lambda row: pd.Series(getGeodetic(row["X"], row["Y"], row["Z"], row["Covariance"])[0]), axis=1
)

	df["Error"] = df.apply(
		lambda row: getGeodetic(row["X"], row["Y"], row["Z"], row["Covariance"])[1], axis=1
	)
	return df
	


def SavePickleData():
    df = generate_Dataframe_vector_xyz('data')

    #filtered_sorted_df = df[df['Station'] == "BABH"].sort_values(by="Date")



    #print(filtered_sorted_df)
    df = convert_to_columns_xyz(df)

    df = convert_geodetic(df)

    save_pickle_Dataframe(df, r'pickle_geodetic_data_columns_xyz.pkl')


def GetSavedData():
    df = pd.read_pickle(r'output\\pickle_geodetic_data_columns_xyz.pkl')

    #print(df)
    return df

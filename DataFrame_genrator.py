import os
import re
import pandas as pd
from FileReader import GetDataFrame

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
	df = df.drop(columns=['arrays']).join(df_expanded)
	return df


def save_pickle_Dataframe(df):
	df.to_pickle('data.pkl')
	

def save_csv_Dataframe(df):
	df.to_csv('output.csv')
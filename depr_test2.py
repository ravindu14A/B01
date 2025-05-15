import pandas as pd
from intermediate import PCA_fit
from utils import draw_graphs
from datetime import datetime

df = pd.read_pickle(r'output\velocity_corrected.pkl')

stations = df['station'].unique()
print(len(stations))


print("on date of the apocalypse:", df[df['date'] == datetime(2004, 12, 26)])


nice_stations = []
for station_name in stations:
	if len(df[df['station'] == station_name]) > 200:
		nice_stations.append(station_name)

for station_name in stations:
	draw_graphs.plot_column_for_station(df, station_name, 'pc2')
# print(len(nice_stations))
# input()
# for station_name in nice_stations:
# 	PCA_fit.compute_and_apply_pca_ne(df, station_name)
# 	draw_graphs.plot_column_for_station(df, station_name, 'pc2')


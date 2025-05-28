import pandas as pd
from utils import df_utils, draw_graphs
from intermediate import PCA_fit
from datetime import datetime
from analysis import outlier_detection, earthquake_correction, fitting
from analysis import model_apply
import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)



df = pd.read_pickle(r'output\intermediate.pkl')

# for station_name in df_utils.get_all_station_names(df):
# 	PCA_fit.compute_and_apply_pca_ne_station(df, station_name)

#print(df.columns)

filtered_stations = sorted(df_utils.get_stations_with_min_data(df, min_points_pre=50, min_points_after=50))
#print(filtered_stations)


filtered_stations = ["ARAU","BEHR","BNKK","GETI", "KMIT","KTPK","KUAL","KUAN","NTUS","PHKT","PHUK","USMP",]




filtered_stations = [ "ARAU","PHUK","PHKT"] 
'''cursed_station = "IPOH"
#from scipy.stats import chi2

#print(chi2.ppf(0.95, df=2))
df = outlier_detection.remove_pre_eq_outliers_en_station(df, cursed_station)
draw_graphs.plot_column_for_station(df, cursed_station, 'd_east_mm')
'''


#filtered_stations = ["KMIT"]

dayssince2005eq = (datetime(2005,3,28) - datetime(2004,12,26)).days
dayssince2012eq = (datetime(2012, 4, 11) - datetime(2004,12,26)).days
print(dayssince2005eq)

for station_name in filtered_stations:
	df = outlier_detection.remove_pre_eq_outliers_en_station(df, station_name)
	df, eigs = PCA_fit.compute_and_apply_pca_ne_station(df, station_name, max_day = 365)#(datetime(2005,3,28) - datetime(2004,12,26)).days
	if eigs is not None: 
		df = outlier_detection.remove_extreme_outliers(df, 'pc1')

		min_start_date = df[df['station']==station_name]['days_since_eq'].min()
		v, d = fitting.fit_station_linear_trend(df, station_name, 'pc1', min_start_date, 0)

		print("v", v, "station", station_name)
		df = earthquake_correction.correct_earthquake_signal_simple_shift(df, station_name, 'pc1', dayssince2005eq, visualize=True)
		
		df = earthquake_correction.correct_earthquake_signal_curve_fitting(df, station_name, 'pc1', dayssince2012eq, visualize=True, v=v)

		df = outlier_detection.remove_post_eq_outliers_pc1_station(df,station_name,v,eigs)


		max_end_date = df[df['station']==station_name]['days_since_eq'].max()
		popt, covt = fitting.fit_station_exponential_decay(df,station_name, 0, max_end_date, 'pc1', v=v, simplified=True)
		print(popt)

		popt, covt, days = model_apply.fit_model_and_find_zero(df, station_name, 'pc1', v)

		print('exponential stuff parameters', popt)
		#print("cov matrix parameters", covt)

		draw_graphs.plot_column_for_station(df, station_name, 'pc1', v,popt)
		if days is not None:
			print(days/365.25, "years", station_name)
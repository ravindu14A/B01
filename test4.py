import pandas as pd
from utils import df_utils, draw_graphs
from intermediate import PCA_fit
from datetime import datetime
from analysis import outlier_detection, earthquake_correction, fitting2
from analysis import model_apply
import numpy as np
np.set_printoptions(precision=3, suppress=True)


df = pd.read_pickle(r'output\intermediate.pkl')

station_name = 'PHKT'

df, eigs = PCA_fit.compute_and_apply_pca_ne_station(df, station_name, max_day = 365)#(datetime(2005,3,28) - datetime(2004,12,26)).days

draw_graphs.plot_column_for_station(df, station_name, 'pc1')


#double exponential (NO v)
# popt, pcov, full_output, mesg, ier = fitting2.fit_station_with_model(df, station_name, 'pc1'
# 								, 1, 365*20
# 								, fitting2.double_exponential, fit_jacob=fitting2.jacobian_double_exponential, initial_guess=None
# 								, bounds= ([0,0,0,0,-np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
# 								)
# print(popt)
# print(pcov)
# print(np.sqrt(np.mean(full_output["fvec"]**2)))
# print(mesg)
# print(ier)

# #single exponential (NO v)
# popt, pcov, full_output, mesg, ier = fitting2.fit_station_with_model(df, station_name, 'pc1'
# 								, 1, 365*20
# 								, fitting2.single_exponential, fit_jacob=fitting2.jacobian_single_exponential, initial_guess=None
# 								)
# print(popt)
# print(pcov)
# print(np.sqrt(np.mean(full_output["fvec"]**2)))
# print(type(full_output["fvec"]))
# print(mesg)
# print(ier)

dayssince2012eq = (datetime(2012, 4, 11) - datetime(2004,12,26)).days
decades_since_2012eq = dayssince2012eq / 365.0 / 10.0
dayssince2005eq = (datetime(2005,3,28) - datetime(2004,12,26)).days
decades_since_2005eq = dayssince2005eq / 365.0 / 10.0



print('eartquake experimentation stuffs')
#earthquake experimentation

#WARNING RESULT IS A FUNCTION TAKING IN DECADES SINCE EQ AND OUTPUTTING CM
popt, pcov, full_output, mesg, ier = fitting2.fit_station_with_model(df, station_name, 'pc1'
								, 1, 365*20
								, fitting2.single_exponential_with_earthquakes, fit_jacob=fitting2.jacobian_single_exponential_with_earthquakes
								, initial_guess=[1,1,1,1]# if doing stuff with earthquakes, intial guess should reflect number of parameters to be optimized, if still breaks consider giving bounds
								, earthquake_dates = [decades_since_2005eq])
print(popt)
print(np.sqrt(np.mean(full_output["fvec"]**2)))
print(pcov)
print(mesg)

# popt, pcov = fitting2.fit_station_with_model(df, station_name, 'pc1'
# 								, 1, 365
# 								, fitting2.single_exponential_v, fit_jacob=fitting2.jacobian_single_exponential_v
# 								, v=.001)



#print(fitting2.single_exponential_with_v_term(3*356, *popt, v=1))

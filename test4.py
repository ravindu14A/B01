import pandas as pd
from utils import df_utils, draw_graphs
from intermediate import PCA_fit
from datetime import datetime
from analysis import outlier_detection, earthquake_correction, fitting2
from analysis import model_apply
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from analysis.datadropper2 import drop_days_after_events
#np.set_printoptions(precision=1)


df = pd.read_pickle(r'output\intermediate.pkl')

station_name = 'KTPK'

df, eigs = PCA_fit.compute_and_apply_pca_ne_station(df, station_name, max_day = 365)#(datetime(2005,3,28) - datetime(2004,12,26)).days
draw_graphs.plot_column_for_station(df, station_name, 'pc1')

dayssince2012eq = (datetime(2012, 4, 11) - datetime(2004,12,26)).days
decades_since_2012eq = dayssince2012eq / 365.0 / 10.0
dayssince2005eq = (datetime(2005,3,28) - datetime(2004,12,26)).days
decades_since_2005eq = dayssince2005eq / 365.0 / 10.0


eq_dates_decades = [dayssince2005eq]


df = drop_days_after_events(df,[ dayssince2012eq], 30*9)

draw_graphs.plot_column_for_station(df, station_name, 'pc1')

print(df[df['station']==station_name]['enu_covariance_cm2'].iloc[50])
print(df[df['station']==station_name]['pca_covariance_pc_space'].iloc[50])

#input()

print('eartquake experimentation stuffs')
#earthquake experimentation

# popt, pcov, _, _, _ = fitting2.fit_station_with_model(df, station_name, 'pc1',
# 								-365*20, 0,
# 								fitting2.linear_model, fit_jacob=fitting2.jacobian_linear_model)

# v,_  = popt


#WARNING RESULT IS A FUNCTION TAKING IN DECADES SINCE EQ AND OUTPUTTING CM CAREFUL WHEN GIVING IT PARAMETERS
# 2 earthquakes
# popt, pcov, full_output, mesg, ier = fitting2.fit_station_with_model(df, station_name, 'pc1'
# 								, 1, 365*20
# 								, fitting2.double_exponential_with_earthquakes_v, fit_jacob=fitting2.jacobian_double_exponential_with_earthquakes_v
# 								, initial_guess=[1,1,1,1,1,1,1]# if doing stuff with earthquakes, intial guess should reflect number of parameters to be optimized, if it still breaks consider giving bounds
# 								, bounds = ([0,0,0,0,-np.inf,-np.inf,-np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
# 								,earthquake_dates = eq_dates_decades, v = v)

#1 earthquake
# popt, pcov, full_output, mesg, ier = fitting2.fit_station_with_model(df, station_name, 'pc1'
# 								, 1, 365*20
# 								, fitting2.double_exponential_with_earthquakes_v, fit_jacob=fitting2.jacobian_double_exponential_with_earthquakes_v
# 								, initial_guess=[1,1,1,1,1,1]# if doing stuff with earthquakes, intial guess should reflect number of parameters to be optimized, if it still breaks consider giving bounds
# 								, bounds = ([0,0,0,0,-np.inf,-np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
# 								,earthquake_dates = [decades_since_2012eq], v = v)
#one earthquake
# popt, pcov, full_output, mesg, ier = fitting2.fit_station_with_model(df, station_name, 'pc1'
# 								, -365*20, 365*20
# 								, fitting2.double_exponential_with_earthquakes_and_linear_region, fit_jacob=fitting2.jacobian_double_exponential_with_earthquakes_and_linear_region
# 								, initial_guess=[1,1,1,1,1,1,1]# if doing stuff with earthquakes, intial guess should reflect number of parameters to be optimized, if it still breaks consider giving bounds
# 								, bounds = ([0,0,0,0,-np.inf,0,-np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
# 								,earthquake_dates = [decades_since_2012eq])

#two earthquakes
popt, pcov, full_output, mesg, ier = fitting2.fit_station_with_model(df, station_name, 'pc1'
								, -365*20, 365*20
								, fitting2.double_exponential_with_earthquakes_and_linear_region, fit_jacob=fitting2.jacobian_double_exponential_with_earthquakes_and_linear_region
								, initial_guess=[1,1,1,1,1,1,1]# if doing stuff with earthquakes, intial guess should reflect number of parameters to be optimized, if it still breaks consider giving bounds
								, bounds = ([0,0,0,0,-np.inf,0,-np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
								,earthquake_dates = eq_dates_decades)

a = popt
[c1, m1, c2, m2, d, v] = a[:6]


print(m1,m2, "exponential coefficients are")
#print(popt)
print("variance is", np.sqrt(np.mean(full_output["fvec"]**2)))


def model_func(x):
	return fitting2.double_exponential_v(x, c1, m1, c2, m2, d, v)

def plot_function(f, start, end, num_points=1000):
	x = np.linspace(start, end, num_points)
	y = f(x)
	plt.plot(x, y, label=f.__name__)
	plt.xlabel('x')
	plt.ylabel('f(x)')
	plt.title(f"Plot of {f.__name__} from {start} to {end}")
	plt.grid(True)
	plt.legend()
	plt.show()

plot_function(model_func, 0, 50)

root = optimize.brentq(model_func, 1, 500000)
print("root is",root) 


print(popt)
print(np.sqrt(np.mean(full_output["fvec"]**2)))
print(pcov)
print(mesg)



import matplotlib.pyplot as plt

def plot_propagated_error(func, jacobian_func, params, pcov, x_plot, 
                         confidence_levels=[1, 2], alpha=0.3, 
                         func_args=(), func_kwargs={}, 
                         jacobian_args=(), jacobian_kwargs={},
                         plot_data=None):
    """
    Plot function with propagated parameter uncertainty bands using analytical Jacobian.
    
    Parameters:
    -----------
    func : callable
        Function f(x, *params, *func_args, **func_kwargs)
    jacobian_func : callable
        Function that returns Jacobian matrix J[i,j] = ∂f(x[i])/∂param[j]
        Should have signature: jacobian_func(x, *params, *jacobian_args, **jacobian_kwargs)
        Returns array of shape (len(x), len(params))
    params : array_like
        Optimized parameters
    pcov : array_like
        Parameter covariance matrix from curve_fit
    x_plot : array_like
        X values for plotting and uncertainty calculation
    confidence_levels : list, optional
        Confidence levels in units of sigma (default: [1, 2] for 1σ, 2σ)
    alpha : float, optional
        Transparency of uncertainty bands
    func_args : tuple, optional
        Additional positional arguments for func
    func_kwargs : dict, optional
        Additional keyword arguments for func
    jacobian_args : tuple, optional
        Additional positional arguments for jacobian_func
    jacobian_kwargs : dict, optional
        Additional keyword arguments for jacobian_func
    plot_data : tuple (x_data, y_data), optional
        Original data points to overlay on plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    x_plot : array of x values used for plotting
    y_pred : array of predicted y values
    sigma_pred : array of prediction uncertainties
    """
    
    # Calculate predictions
    y_pred = func(x_plot, *params, *func_args, **func_kwargs)
    
    # Calculate Jacobian matrix at all x points
    jacobian_matrix = jacobian_func(x_plot, *params, *jacobian_args, **jacobian_kwargs)
    
    # Calculate prediction uncertainty for each x point
    # For each row i: σ²_pred[i] = J[i,:] @ pcov @ J[i,:].T
    sigma_pred = np.zeros(len(x_plot))
    
    for i in range(len(x_plot)):
        gradient = jacobian_matrix[i, :]  # Gradient at x_plot[i]
        sigma_pred[i] = np.sqrt(gradient @ pcov @ gradient.T)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original data if provided
    if plot_data is not None:
        x_data, y_data = plot_data
        ax.scatter(x_data, y_data, color='red', alpha=0.7, s=30, 
                   label='Data points', zorder=5)
    
    # Plot best fit
    ax.plot(x_plot, y_pred, 'b-', linewidth=2, label='Best fit', zorder=4)
    
    # Plot confidence bands
    colors = ['lightblue', 'lightgray', 'lightyellow']
    for i, sigma_level in enumerate(confidence_levels):
        color = colors[i % len(colors)]
        
        upper_bound = y_pred + sigma_level * sigma_pred
        lower_bound = y_pred - sigma_level * sigma_pred
        
        ax.fill_between(x_plot, lower_bound, upper_bound, 
                       alpha=alpha, color=color, 
                       label=f'{sigma_level}σ confidence', zorder=i+1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Function Fit with Propagated Parameter Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    return fig, ax, x_plot, y_pred, sigma_pred


xplot = np.linspace(0, 30, 10000)
plot_propagated_error(
	fitting2.double_exponential_with_earthquakes_and_linear_region, fitting2.jacobian_double_exponential_with_earthquakes_and_linear_region
    , popt, pcov, xplot
    , func_kwargs={"earthquake_dates": eq_dates_decades},jacobian_kwargs={"earthquake_dates": eq_dates_decades}
)
# popt, pcov = fitting2.fit_station_with_model(df, station_name, 'pc1'
# 								, 1, 365
# 								, fitting2.single_exponential_v, fit_jacob=fitting2.jacobian_single_exponential_v
# 								, v=.001)



#print(fitting2.single_exponential_with_v_term(3*356, *popt, v=1))
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
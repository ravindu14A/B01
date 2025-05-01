import numpy as np
import pandas as pd
import pickle
from preprocessing.internal_dataclass.dataset import Station, GeoDataset
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.optimize import fsolve


def custom_func(x, a, b, c, d, e, f):
    return x * a + x**2 * b + x**3 * c + x**4 * d + e


#import the data as a dataframe for ease of use
#define longtitide and latitude as long and lat
def combine(df):
    df.reset_index(inplace=True)
    #loop through data frame and find their distance as a combination of longtitude and latitude
    df['dist'] = np.sqrt(df['long']**2 + df['lat']**2)

    # Assume the big jump is on the 20th of december
    '''df['dist_diff'] = df['dist'].diff()
    jump_index = df['dist_diff'].abs().idxmax()
    jump_date = df.loc[jump_index, 'date']'''

    jump_date = pd.to_datetime('2004-12-01')
    df_before_jump = df[df['date'] < jump_date]

    # Step 3: Calculate slope using linear regression
    # Make sure date is converted to numerical format
    df['date_ordinal'] = pd.to_datetime(df['date']).map(pd.Timestamp.toordinal)
    df_before_jump['date_ordinal'] = pd.to_datetime(df_before_jump['date']).map(pd.Timestamp.toordinal)
    slope, intercept, r_value, p_value, std_err = linregress(df_before_jump['date_ordinal'], df_before_jump['dist'])

    # Step 1: Get trend line for dates before the jump
    df['trend'] = slope * df['date_ordinal'] + intercept

    # Step 2: Subtract the trend from original dist to get detrended values
    df['dist_detrended'] = df['dist'] - df['trend']

    # Find the intercept
    start_date = pd.to_datetime('2005-01-01')
    df_after_2005 = df[df['date'] >= start_date].copy()

    # Fit the polynomial
    x = df_after_2005['date_ordinal'].values
    y = df_after_2005['dist_detrended'].values  # or 'dist' if you didn’t detrend


    popt, pcov = curve_fit(custom_func, x, y, p0=[0.4, 0.01, 0.1, 0.5, 0.5, 0.2])  # p0 = initial guess
    y_fit = custom_func(x, *popt)


    # Define a zero-crossing finder near the midpoint
    # Evaluate the function over a dense range
    x_dense = np.linspace(x.min(), x.max() + 365_000, 5000)
    y_dense = custom_func(x_dense, *popt)
    print(f"Min y: {y_dense.min()}, Max y: {y_dense.max()}")
    # Find sign changes (where y crosses zero)
    sign_changes = np.where(np.diff(np.sign(y_dense)))[0]

    # Initialize list of roots
    roots = []

    for idx in sign_changes:
        x0 = x_dense[idx]  # left bound of sign change
        x1 = x_dense[idx + 1]  # right bound of sign change

        # Use midpoint as initial guess
        root = fsolve(lambda t: custom_func(t, *popt), x0)[0]

        if x.min() <= root <= x.max() + 365_000:
            roots.append(root)

    # Convert to readable dates
    intercept_dates = [pd.Timestamp.fromordinal(int(r)) for r in roots]

    # Print the dates
    for i, date in enumerate(intercept_dates, 1):
        print(f"X-axis Intercept {i}: {date.date()}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df_after_2005['date'], y, label='Detrended Data', alpha=0.6)
    plt.plot(df_after_2005['date'], y_fit, label='Fitted Function', linestyle='--', linewidth=2)



    # Labels and legend
    plt.xlabel('Date')
    plt.ylabel('Detrended Displacement')
    plt.title('Fitted Curve and X-Axis Intercept')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''# Fit a polynomial of degree 2 (quadratic)
    degree = 2
    coeffs = np.polyfit(x, y, degree)

    # Create polynomial function
    poly_func = np.poly1d(coeffs)

    # Find real roots (x-intercepts in ordinal)
    roots = np.roots(coeffs)
    real_roots = [r for r in roots if np.isreal(r)]
    real_roots = np.real(real_roots)

    intercept_dates = []
    for r in real_roots:
        if r > 0:
            # Convert roots to dates
            intercept_dates.append(pd.to_datetime(pd.Timestamp.fromordinal(int(r))))

    # Print the results
    for i, date in enumerate(intercept_dates, 1):
        print(f"X-axis Intercept {i}: {date.date()}")

    # Apply it to get fitted values
    df_after_2005['poly_fit'] = poly_func(x)
'''


    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['long'], label='Column 1')
    plt.plot(df['date'], df['lat'], label='Column 2')
    plt.plot(df['date'], df['dist_detrended'], label='Column 3')

    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Date vs Columns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


with open('KUAL_new.pkl', 'rb') as f:
    station_obj = pickle.load(f)
station_obj.data.drop('index', axis=1, inplace=True)
print(station_obj.data)

combine(station_obj.data)
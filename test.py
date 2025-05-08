from analysis import earthquake_correction
import pandas as pd
import draw_graphs
from analysis import model_stuff
from analysis import curve_fit

df = pd.read_pickle(r'output\velocity_corrected.pkl')


station_name = "ARAU"
columntobecalced = 'd_north_mm'

df = earthquake_correction.remove_earthquake_column(df, station_name, '2012-04-11',columntobecalced)
df = earthquake_correction.remove_earthquake_column(df, station_name, 'March 28, 2005',columntobecalced)

draw_graphs.plot_displacement_for_station(df,station_name)

popt, pcov, ref_date, v = model_stuff.find_curve_for_station(df, station_name, columntobecalced)

print(popt)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from analysis.curve_fit import model_with_known_v  # assumes you're plotting full model

def plot_displacement_for_station(df, station_name, column, v=None, popt=None, ref_date=None):
    # Filter DataFrame by station
    station_data = df[df['station'] == station_name].copy()
    station_data['date'] = pd.to_datetime(station_data['date'])

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(station_data['date'], station_data[column], color='b', linewidth=1, label='Observed')

    # Plot model if parameters provided
    if popt is not None and ref_date is not None:
        quake_date = pd.Timestamp("2004-12-26")
        end_model_date = quake_date + pd.DateOffset(years=200)

        # Generate daily dates from quake_date to 200 years after
        model_dates = pd.date_range(start=quake_date, end=end_model_date, freq='D')
        x_model = (model_dates - ref_date).days
        y_model = model_with_known_v(x_model, *popt, v)

        ax.plot(model_dates, y_model, color='orange', linestyle='--', linewidth=2, label='Fitted Model (post-2004)')

    # Set Y-axis limit to 100mm
    ax.set_ylim(-150, 20)  # Limit Y-axis to 100mm

    # Labels and formatting
    ax.set_ylabel(column.replace('d_', '').replace('_mm', '').capitalize() + ' (mm)')
    ax.set_title(f"{column} Displacement for Station {station_name}")
    ax.grid(True)

    # Earthquake vertical lines
    df_earthquakes = pd.read_pickle(r'raw_data\earthquakes_records')
    if not df_earthquakes.empty:
        for date in df_earthquakes['date']:
            ax.axvline(date, color='k', linestyle='--', linewidth=1)

    # Format x-axis as years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Date')
    ax.legend()

    plt.tight_layout()
    plt.show()


plot_displacement_for_station(df, station_name, columntobecalced,v =v, popt=popt, ref_date=ref_date)

from scipy.optimize import root_scalar

from scipy.optimize import root_scalar
from analysis.curve_fit import model_with_known_v
import pandas as pd

def find_height_reintercept_date(popt, v, ref_date, height_intercept):
    # Define f(x) = model(x) - height_intercept
    def func(x):
        return model_with_known_v(x, *popt, v) - height_intercept

    # Check function at bracket endpoints
    func_at_start = func(0)
    func_at_end = func(365 * 10000)  # Change to 10,000 years
    print(f"func(0): {func_at_start}, func(365*10000): {func_at_end}")

    # If the signs at the ends are the same, root-finding won't work, so adjust the bounds
    if func_at_start * func_at_end > 0:
        print("Function doesn't change sign within the initial bracket. Adjusting bounds.")
        return None  # Or adjust the bounds manually if needed

    # Try to solve in range [0, 365*10000] = 10,000 years
    sol = root_scalar(func, bracket=[0, 365 * 10000], method='brentq')
    if not sol.converged:
        raise RuntimeError("Could not find re-intercept within 10,000 years.")

    # Convert days to years after ref_date
    years_after_ref = sol.root / 365  # Convert days into years

    print(f"Model height re-intercepts {years_after_ref:.2f} years after the reference date.")

    return years_after_ref

print(find_height_reintercept_date(popt,v,ref_date,20))
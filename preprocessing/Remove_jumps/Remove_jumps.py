import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

with open('BEHR.pkl', 'rb') as f:
    station_obj = pickle.load(f)


def latlong_to_cm(df):

    lat0 = df['lat'].iloc[0]
    lon0 = df['long'].iloc[0]

    # Earth's constants
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * np.cos(np.radians(lat0))

    # Calculate deltas in meters
    delta_lat_m = (df['lat'] - lat0) * meters_per_deg_lat
    delta_lon_m = (df['long'] - lon0) * meters_per_deg_lon

    # Convert to centimeters
    df['x_cm'] = delta_lon_m * 100
    df['y_cm'] = delta_lat_m * 100

    return df


def remove_jumps(df, start_date, end_date, jump_threshold_cm=.05):
    """
    Removes discontinuities in the 'x_cm' and 'y_cm' between the given date range.
    Args:
        df (pd.DataFrame): must have 'date', 'x_cm', 'y_cm' columns.
        start_date (str or pd.Timestamp): start date of the range.
        end_date (str or pd.Timestamp): end date of the range.
        jump_threshold_cm (float): size of jump in cm to consider as discontinuity.
    Returns:
        pd.DataFrame: corrected DataFrame.
    """
    # Ensure date is in datetime format
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Find starting and ending indices
    start_idx = df[df['date'] >= start_date].index[0]
    end_idx = df[df['date'] <= end_date].index[-1]

    # Initialize shift amounts
    shift_x = 0
    shift_y = 0

    # Create shifted x and y columns
    corrected_x = df['x_cm'].copy()
    corrected_y = df['y_cm'].copy()

    # Loop through the data within the date range
    for i in range(start_idx + 1, end_idx + 1):
        delta_x = (df['x_cm'][i] - df['x_cm'][i - 1])
        delta_y = (df['y_cm'][i] - df['y_cm'][i - 1])

        if np.abs(delta_x) > jump_threshold_cm or np.abs(delta_y) > jump_threshold_cm:
            print(df['date'].iloc[i], df['x_cm'].iloc[i], df['y_cm'].iloc[i], delta_x, delta_y)
            # Detected a jump
            shift_x += -delta_x
            shift_y += -delta_y

        # Apply current shift
        corrected_x[i] += shift_x
        corrected_y[i] += shift_y

    # Update the DataFrame
    df['x_cm_corrected'] = corrected_x
    df['y_cm_corrected'] = corrected_y

    return df




# Apply the transformation
station_obj = latlong_to_cm(station_obj)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot x_cm vs date
axs[0].plot(station_obj['date'], station_obj['x_cm'], marker='o', linestyle='-')
axs[0].set_ylabel('X (cm)')
axs[0].set_title('X Position over Time')
axs[0].grid(True)

# Plot y_cm vs date
axs[1].plot(station_obj['date'], station_obj['y_cm'], marker='o', linestyle='-', color='orange')
axs[1].set_ylabel('Y (cm)')
axs[1].set_xlabel('Date')
axs[1].set_title('Y Position over Time')
axs[1].grid(True)

plt.tight_layout()
plt.show()

corrected_station_obj = remove_jumps(station_obj, start_date='2010-01-01', end_date='2020-01-01', jump_threshold_cm=.05)

# Plot corrected data
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(corrected_station_obj['date'], corrected_station_obj['x_cm_corrected'], label='Corrected X', marker='o')
axs[0].set_ylabel('X (cm)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(corrected_station_obj['date'], corrected_station_obj['y_cm_corrected'], label='Corrected Y', marker='o', color='orange')
axs[1].set_ylabel('Y (cm)')
axs[1].set_xlabel('Date')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def local_outliers(x, y, num_segments=5, threshold=1):
    outlier_mask = np.zeros_like(y, dtype=bool)
    trend_line = np.zeros_like(y)
    residuals = np.zeros_like(y)
    
    indices = np.arange(len(y))
    segments = np.array_split(indices, num_segments)
    
    for seg in segments:
        # Fit linear trend on the segment
        coeffs = np.polyfit(x[seg], y[seg], 1)
        trend_seg = np.polyval(coeffs, x[seg])
        trend_line[seg] = trend_seg
        # Compute residuals in the segment
        residual_seg = y[seg] - trend_seg
        residuals[seg] = residual_seg
        # Compute robust stats for residuals in the segment
        median_res = np.median(residual_seg)
        mad_res = np.median(np.abs(residual_seg - median_res))
        scaled_mad = 1.4826 * mad_res
        # Prevent division by zero if no variation in the segment
        if scaled_mad == 0:
            seg_outliers = np.zeros_like(residual_seg, dtype=bool)
        else:
            seg_outliers = np.abs(residual_seg - median_res) > threshold * scaled_mad
        outlier_mask[seg] = seg_outliers
        
    return trend_line, residuals, outlier_mask

# Load DataFrame and ensure 'date' is datetime
df = pd.read_pickle("J001.pkl")
df['date'] = pd.to_datetime(df['date'])

# Convert date to numeric values (ordinal) for trend fitting
x = df['date'].map(pd.Timestamp.toordinal).values

# ----- Latitude: Local detrending with 5 segments -----
y_lat = df['latitude'].values
lat_trend, lat_residuals, lat_outlier = local_outliers(x, y_lat, num_segments=5, threshold=3)

# ----- Longitude: Local detrending with 5 segments -----
y_long = df['longitude'].values
long_trend, long_residuals, long_outlier = local_outliers(x, y_long, num_segments=5, threshold=3)

# ----- Height: Global detrending (since height is supposed to grow over time) -----
y_height = df['height'].values
coeffs_height = np.polyfit(x, y_height, 1)
trend_height = np.polyval(coeffs_height, x)
height_residuals = y_height - trend_height
median_height_res = np.median(height_residuals)
mad_height_res = np.median(np.abs(height_residuals - median_height_res))
scaled_mad_height = 1.4826 * mad_height_res
height_outlier = np.abs(height_residuals - median_height_res) > 3 * scaled_mad_height

# Combine overall outlier mask: flag a row if any variable is an outlier
combined_outlier_mask = lat_outlier | long_outlier | height_outlier

# ----- Earthquake Event Detection -----
# Define the window size and the outlier threshold for an event.
window_size = 30
event_threshold = 6

# Compute the rolling sum of outliers (convert boolean to int)
rolling_sum = np.convolve(combined_outlier_mask.astype(int), np.ones(window_size, dtype=int), mode='valid')

# Find windows where the number of outliers is at least 100
event_windows = np.where(rolling_sum >= event_threshold)[0]
if len(event_windows) > 0:
    # Use the first window found and take the midpoint as the representative index.
    first_event_index = event_windows[0] + window_size // 2
    earthquake_date = df['date'].iloc[first_event_index]
    print("First earthquake event detected at index", first_event_index, "with date", earthquake_date)
else:
    earthquake_date = None
    print("No earthquake event detected.")

# Create a cleaned DataFrame with non-outlier rows
df_clean = df[~combined_outlier_mask]
df_clean.to_pickle("J001_cleaned.pkl")
print("Cleaned DataFrame saved to J001_cleaned.pkl")

# ----- Plotting -----
# Create a 3x2 grid:
# For Latitude and Longitude: left plots show data with local trend, right plots show residuals.
# For Height: left plot shows global trend, right plot shows residuals.
fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True)

# Latitude plots:
axes[0,0].scatter(df['date'], df['latitude'], label='Data', color='blue', s=20)
axes[0,0].plot(df['date'], lat_trend, label='Local Trend', color='green')
axes[0,0].scatter(df.loc[lat_outlier, 'date'], df.loc[lat_outlier, 'latitude'], 
                   label='Outlier', facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[0,0].set_ylabel('Latitude')
axes[0,0].set_title('Latitude vs Date with Local Trend & Outliers')
axes[0,0].legend()

axes[0,1].scatter(df['date'], lat_residuals, label='Residuals', color='blue', s=20)
axes[0,1].scatter(df.loc[lat_outlier, 'date'], lat_residuals[lat_outlier],
                   label='Outlier Residuals', facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[0,1].axhline(np.median(lat_residuals), color='green', label='Median Residual')
axes[0,1].set_ylabel('Latitude Residuals')
axes[0,1].set_title('Latitude Residuals vs Date')
axes[0,1].legend()

# Longitude plots:
axes[1,0].scatter(df['date'], df['longitude'], label='Data', color='blue', s=20)
axes[1,0].plot(df['date'], long_trend, label='Local Trend', color='green')
axes[1,0].scatter(df.loc[long_outlier, 'date'], df.loc[long_outlier, 'longitude'], 
                   label='Outlier', facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[1,0].set_ylabel('Longitude')
axes[1,0].set_title('Longitude vs Date with Local Trend & Outliers')
axes[1,0].legend()

axes[1,1].scatter(df['date'], long_residuals, label='Residuals', color='blue', s=20)
axes[1,1].scatter(df.loc[long_outlier, 'date'], long_residuals[long_outlier],
                   label='Outlier Residuals', facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[1,1].axhline(np.median(long_residuals), color='green', label='Median Residual')
axes[1,1].set_ylabel('Longitude Residuals')
axes[1,1].set_title('Longitude Residuals vs Date')
axes[1,1].legend()

# Height plots:
axes[2,0].scatter(df['date'], df['height'], label='Data', color='blue', s=20)
axes[2,0].plot(df['date'], trend_height, label='Global Trend', color='green')
axes[2,0].scatter(df.loc[height_outlier, 'date'], df.loc[height_outlier, 'height'], 
                   label='Outlier', facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[2,0].set_ylabel('Height')
axes[2,0].set_title('Height vs Date with Global Trend & Outliers')
axes[2,0].legend()

axes[2,1].scatter(df['date'], height_residuals, label='Residuals', color='blue', s=20)
axes[2,1].scatter(df.loc[height_outlier, 'date'], height_residuals[height_outlier],
                   label='Outlier Residuals', facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[2,1].axhline(np.median(height_residuals), color='green', label='Median Residual')
axes[2,1].set_ylabel('Height Residuals')
axes[2,1].set_title('Height Residuals vs Date')
axes[2,1].legend()

# If an earthquake event is detected, mark it with a vertical dashed magenta line on all subplots.
if earthquake_date is not None:
    for ax in axes.flatten():
        ax.axvline(earthquake_date, color='magenta', linestyle='--', linewidth=2, label='Earthquake Event')
    # To avoid duplicate labels in legends, adjust the legend for one subplot.
    axes[0,0].legend()

plt.tight_layout()
plt.show()

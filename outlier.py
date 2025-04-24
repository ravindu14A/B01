import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Modified local_outliers function ---
def local_outliers(x, y, num_segments, threshold):
    outlier_mask = np.zeros_like(y, dtype=bool)
    trend_line = np.zeros_like(y)
    residuals = np.zeros_like(y)

    indices = np.arange(len(y))
    segments = np.array_split(indices, num_segments)  # Split indices into segments

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
        if scaled_mad == 0:
            seg_outliers = np.zeros_like(residual_seg, dtype=bool)
        else:
            seg_outliers = np.abs(residual_seg - median_res) > threshold * scaled_mad
        outlier_mask[seg] = seg_outliers

    return trend_line, residuals, outlier_mask, segments

# --- Reading Data ---
#df = pd.read_csv("C:/Users/potfi/Documents/GitHub/bom/B01/preprocessing/processed_data/SE_Asia/ARAU.txt")
df = pd.read_csv("C:/Users/potfi/Documents/GitHub/bom/B01/preprocessing/processed_data/SE_Asia/BABH.txt")
df = df.rename(columns={'Date': 'date', 'lat': 'latitude', 'long': 'longitude', 'alt': 'height'})
df['date'] = pd.to_datetime(df['date'])

# Convert date to numeric values (ordinal) for trend fitting
x = df['date'].map(pd.Timestamp.toordinal).values

# --- Latitude: Local detrending with 5 segments ---
y_lat = df['latitude'].values
lat_trend, lat_residuals, lat_outlier, lat_segments = local_outliers(x, y_lat, num_segments=10, threshold=2)

# --- Longitude: Local detrending with 5 segments ---
y_long = df['longitude'].values
long_trend, long_residuals, long_outlier, long_segments = local_outliers(x, y_long, num_segments=10, threshold=2)

# --- Height: Local detrending with 5 segments (now segmented like latitude and longitude) ---
y_height = df['height'].values
height_trend, height_residuals, height_outlier, height_segments = local_outliers(x, y_height, num_segments=10, threshold=2)

# --- Earthquake Event Detection ---
window_size = 150
event_threshold = 100

# Combine overall outlier mask: flag a row if any variable is an outlier
combined_outlier_mask = lat_outlier | long_outlier | height_outlier

# Compute the rolling sum of outliers (convert boolean to int)
rolling_sum = np.convolve(combined_outlier_mask.astype(int), np.ones(window_size, dtype=int), mode='valid')

# Find windows where the number of outliers is at least 100
event_windows = np.where(rolling_sum >= event_threshold)[0]
if len(event_windows) > 0:
    first_event_index = event_windows[0] + window_size // 2
    earthquake_date = df['date'].iloc[first_event_index]
    print("First earthquake event detected at index", first_event_index, "with date", earthquake_date)
else:
    earthquake_date = None
    print("No earthquake event detected.")

# Build final removal mask: remove rows flagged as outliers or within the earthquake event window.
final_removal_mask = combined_outlier_mask.copy()
if earthquake_date is not None:
    event_start = event_windows[0]
    event_indices = np.arange(event_start, min(event_start + window_size, len(df)))
    final_removal_mask[event_indices] = True

df_clean = df[~final_removal_mask]
df_clean.to_pickle("ARAU_cleaned.pkl")
print("Cleaned DataFrame saved to ARAU_cleaned.pkl")

# --- Plotting: Data parsed by segments ---
fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True)

# Latitude Plots:
for seg in lat_segments:
    axes[0,0].scatter(df['date'].iloc[seg], df['latitude'].iloc[seg], color='blue', s=20)
    axes[0,0].plot(df['date'].iloc[seg], lat_trend[seg], color='green')
    axes[0,0].scatter(df['date'].iloc[seg][lat_outlier[seg]], df['latitude'].iloc[seg][lat_outlier[seg]],
                       facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[0,0].set_ylabel('Latitude [rad]')
axes[0,0].set_title('Latitude vs Date with Local Trend & Outliers')

for seg in lat_segments:
    axes[0,1].scatter(df['date'].iloc[seg], lat_residuals[seg], color='blue', s=20)
    axes[0,1].scatter(df['date'].iloc[seg][lat_outlier[seg]], lat_residuals[seg][lat_outlier[seg]],
                       facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[0,1].axhline(np.median(lat_residuals), color='green', label='Median Residual')
axes[0,1].set_ylabel('Latitude Residuals')
axes[0,1].set_title('Latitude Residuals vs Date')
axes[0,1].legend()

# Longitude Plots:
for seg in long_segments:
    axes[1,0].scatter(df['date'].iloc[seg], df['longitude'].iloc[seg], color='blue', s=20)
    axes[1,0].plot(df['date'].iloc[seg], long_trend[seg], color='green')
    axes[1,0].scatter(df['date'].iloc[seg][long_outlier[seg]], df['longitude'].iloc[seg][long_outlier[seg]],
                       facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[1,0].set_ylabel('Longitude [rad]')
axes[1,0].set_title('Longitude vs Date with Local Trend & Outliers')

for seg in long_segments:
    axes[1,1].scatter(df['date'].iloc[seg], long_residuals[seg], color='blue', s=20)
    axes[1,1].scatter(df['date'].iloc[seg][long_outlier[seg]], long_residuals[seg][long_outlier[seg]],
                       facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[1,1].axhline(np.median(long_residuals), color='green', label='Median Residual')
axes[1,1].set_ylabel('Longitude Residuals')
axes[1,1].set_title('Longitude Residuals vs Date')
axes[1,1].legend()

# Height Plots:
for seg in height_segments:
    axes[2,0].scatter(df['date'].iloc[seg], df['height'].iloc[seg], color='blue', s=20)
    axes[2,0].plot(df['date'].iloc[seg], height_trend[seg], color='green')
    axes[2,0].scatter(df['date'].iloc[seg][height_outlier[seg]], df['height'].iloc[seg][height_outlier[seg]],
                       facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[2,0].set_ylabel('Height [m]')
axes[2,0].set_xlabel('date [years]')
axes[2,0].set_title('Height vs Date with Local Trend & Outliers')

for seg in height_segments:
    axes[2,1].scatter(df['date'].iloc[seg], height_residuals[seg], color='blue', s=20)
    axes[2,1].scatter(df['date'].iloc[seg][height_outlier[seg]], height_residuals[seg][height_outlier[seg]],
                       facecolors='none', edgecolors='red', s=50, linewidths=1.5)
axes[2,1].axhline(np.median(height_residuals), color='green', label='Median Residual')
axes[2,1].set_ylabel('Height Residuals')
axes[2,1].set_xlabel('date [years]')
axes[2,1].set_title('Height Residuals vs Date')
axes[2,1].legend()

# Mark earthquake event across all subplots, if detected.
if earthquake_date is not None:
    for ax in axes.flatten():
        ax.axvline(earthquake_date, color='magenta', linestyle='--', linewidth=2, label='Earthquake Event')
    axes[0,0].legend()

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

from preprocessing.internal_dataclass.dataset import GeoDataset, Station


class OutlierDetector:
    def __init__(self, dataset: GeoDataset):
        self.dataset = dataset

    @staticmethod
    def __local_outliers(x, y, num_segments=None, threshold=None):
        """
            ...
        """
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

    def _clean_sample(self, sample: Station):
        df = sample.data
        df.reset_index(inplace=True)
        x = df['Date'].map(pd.Timestamp.toordinal).values

        # Latitute
        y_lat = df['lat'].values
        lat_trend, lat_residuals, lat_outlier = self.__local_outliers(x, y_lat, num_segments=5, threshold=3)

        # Longitude
        y_long = df['longitude'].values
        long_trend, long_residuals, long_outlier = self.__local_outliers(x, y_long, num_segments=5, threshold=3)

        # Height
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
        window_size = 150
        event_threshold = 100

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

        # ----- Build final removal mask -----
        # In addition to individual outliers, remove all points in the detected earthquake event window.
        final_removal_mask = combined_outlier_mask.copy()
        if earthquake_date is not None:
            # event_windows[0] is the starting index of the window in the original array.
            event_start = event_windows[0]
            event_indices = np.arange(event_start, min(event_start + window_size, len(df)))
            final_removal_mask[event_indices] = True

        # Create a cleaned DataFrame with non-outlier rows and remove earthquake event points
        df_clean = df[~final_removal_mask]
        print(f"Cleaned DataFrame {sample.name}")

        station = Station(
            name=sample.name,
            position=sample.position,
            data=df_clean,
        )
        return station

    def clean_dataset(self):
        stations = []

        for sample in self.dataset.samples:
            cleaned_sample = self._clean_sample(sample)
            stations.append(cleaned_sample)

        cleaned_dataset = GeoDataset(samples=stations)

        return cleaned_dataset
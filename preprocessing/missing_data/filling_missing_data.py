import pandas as pd
from preprocessing.internal_dataclass.dataset import Station, GeoDataset
import pickle

class MissingDataGNSS:
    """
    A class for methods to handle missing values
    in all three spatial directions (latitude, longitude, and height).
    """

    def __init__(self, dataset: GeoDataset):

        self.dataset = dataset

    def processing_all_files(self):

        filled_samples = []

        for station in self.dataset.samples:
            processed_df = self._process_all_directions(station.data)
            filled_sample = Station(
                name=station.name,
                position=station.position,
                data=processed_df
            )

            filled_samples.append(filled_sample)

        return GeoDataset(samples=filled_samples)





    @staticmethod
    def _fill_missing_dates(data: Station):

        data["Date"] = pd.to_datetime(data["Date"], format="%y%b%d")
        # Generate the full date range
        full_date_range = pd.date_range(
            start=data["Date"].min(),
            end=data["Date"].max(),
            freq='D'
        )
        # Convert to DataFrame
        full_date_df = pd.DataFrame({"Date": full_date_range})
        full_date_df["Date"] = pd.to_datetime(full_date_df["Date"])

        # Merge with original data to create a complete time series
        processed_data = pd.merge(full_date_df, data, on="Date", how="left")

        return processed_data

    @staticmethod
    def _interpolate_missing_values(data, columns=None, method='linear'):

        if columns is None:
            raise ValueError("Columns were not given.")

        processed_data = data.copy()

        # Set date as index for interpolation
        if 'Date' in processed_data.columns:
            processed_data.set_index("Date", inplace=True)

        # Interpolate missing values in each column
        for column in columns:
            if column in processed_data.columns:
                processed_data[column] = processed_data[column].interpolate(method=method)

        return processed_data

    def _process_all_directions(self, data, lat_col='X', lon_col='Y',
                               height_col='Z', method='linear'):


        # Fill missing dates
        data = self._fill_missing_dates(data)

        # Create a list of columns to interpolate
        columns_to_interpolate = []

        for col in [lat_col, lon_col, height_col]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                columns_to_interpolate.append(col)

        processed_data = self._interpolate_missing_values(data, columns=columns_to_interpolate, method=method)

        return processed_data

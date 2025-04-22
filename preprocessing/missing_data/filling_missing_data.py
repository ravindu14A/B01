import pandas as pd


class MissingDataGNSS:
    """
    A class for methods to handle missing values
    in all three spatial directions (latitude, longitude, and height).
    """

    def __init__(self, df):

        self.data = df
        self.processed_data = None


    def fill_missing_dates(self):

        # Generate the full date range
        full_date_range = pd.date_range(
            start=self.data["Date"].min(),
            end=self.data["Date"].max(),
            freq='D'
        )

        # Convert to DataFrame
        full_date_df = pd.DataFrame({"Date": full_date_range})

        # Merge with original data_Thailand to create a complete time series
        self.processed_data = pd.merge(full_date_df, self.data, on="Date", how="left")

        return self.processed_data

    def interpolate_missing_values(self, columns=None, method='linear'):

        if self.processed_data is None:
            raise ValueError("Processed data_Thailand not available. Call fill_missing_dates() first.")

        if columns is None:
            raise ValueError("Columns were not given.")

        # Set date as index for interpolation
        if 'Date' in self.processed_data.columns:
            self.processed_data.set_index("Date", inplace=True)

        # Interpolate missing values in each column
        for column in columns:
            if column in self.processed_data.columns:
                self.processed_data[column] = self.processed_data[column].interpolate(method=method)

        return self.processed_data

    def process_all_directions(self, lat_col='_latitude(deg)', lon_col='_longitude(deg)',
                               height_col='__height(m)', method='linear'):


        # Fill missing dates
        self.fill_missing_dates()

        # Create a list of columns to interpolate
        columns_to_interpolate = []

        for col in [lat_col, lon_col, height_col]:
            if col in self.processed_data.columns:
                columns_to_interpolate.append(col)

        self.interpolate_missing_values(columns=columns_to_interpolate, method=method)

        return self.processed_data

    def save_processed_data(self, output_path):

        if self.processed_data is None:
            raise ValueError("No processed data_Thailand available to save.")

        try:
            # Reset index to include the Date column in the saved file
            data_to_save = self.processed_data.copy()
            if data_to_save.index.name == 'Date':
                data_to_save = data_to_save.reset_index()

            data_to_save.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving data_Thailand: {e}")
            return False

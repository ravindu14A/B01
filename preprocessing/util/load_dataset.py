import os
import pandas as pd
import pickle
from typing import List, Dict
from datetime import datetime
from preprocessing.internal_dataclass.dataset import *


def load_geodataset(stations_dir) -> GeoDataset:
    """
    Load a geospatial dataset from provided configuration, processing station pickle files.

    Returns:
        GeoDataset: A dataset containing Station instances with their observations
    """

    # Get list of pickle files in the stations directory
    station_files = [f for f in os.listdir(stations_dir) if f.endswith('.pkl')]

    samples = []
    # Process each station file
    for station_file in station_files:
        # Extract station name from filename
        station_name = station_file.replace('.pkl', '')

        # Full path to the pickle file
        file_path = os.path.join(stations_dir, station_file)

        try:
            # Load the pickle file containing the station DataFrame
            with open(file_path, 'rb') as f:
                station_df = pickle.load(f)

            # Use default position or first observation if not specified
            station_position = [0.0, 0.0, 0.0]
            if not station_df.empty:
                first_row = station_df.iloc[0]
                station_position = [first_row['X'], first_row['Y'], first_row['Z']]

            # Create a new station
            station = Station(
                name=station_name,
                position=station_position,
                file_path=file_path,
            )

            # Add the station to the dataset
            samples.append(station)


        except Exception as e:
            print(f"Error processing station {station_name}: {str(e)}")

    dataset = GeoDataset(samples=samples)
    return dataset


print(load_geodataset('..\processed_data\SE_Asia'))


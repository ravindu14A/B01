import os
from collections import defaultdict
from datetime import datetime
import re
import pandas as pd

class StationDataProcessor:
    def __init__(self, input_directory, output_directory):
        """
        Initialize the processor with input and output directories.

        Args:
            input_directory (str): Path to the directory containing input files
            output_directory (str): Path to the directory where output files will be saved
        """
        self.input_directory = input_directory
        self.output_directory = output_directory

        '''# Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)'''

    def process_files(self):
        """
        Process all files in the input directory and create station-specific files.

        Returns:
            dict: A dictionary with station names as keys and output file paths as values
        """


        '''# Dictionary to store data for each station
        station_data = defaultdict(list)

        # Get all files in the input directory
        input_files = [f for f in os.listdir(self.input_directory) if
                       os.path.isfile(os.path.join(self.input_directory, f))]

        # Process each file
        for file_name in input_files:
            file_path = os.path.join(self.input_directory, file_name)
            self._process_file(file_path, station_data)

        # Write station data to output files
        output_files = self._write_station_files(station_data)'''


    def _process_file(self):
        input_file = "../data/PFITRF14003.00C"

        with open(input_file, "r") as file:
            context = file.readlines()

        match = re.search(r"(\d{2}[A-Z]{3}\d{2})", context[0])
        date = match.group(1)

        print(match)

        print(context)

        station_dict = defaultdict(lambda: defaultdict(lambda: {"X": None, "Y": None, "Z": None}))


    def _parse_line(self, line):
        ...

    def _write_station_files(self, station_data):
        '''
            Example:
                station_data = {'UHM3': {'2005': [1, 2, 3], '2006': [4, 2, 5]},
                'USHG': {'2003': [5, 6, 7], '2006': [6, 7, 8]},}
        '''

        for station, data in station_data.items():
            # Convert dictionary to DataFrame
            df = pd.DataFrame.from_dict(data, orient='index', columns=['X', 'Y', 'Z'])
            df.index.name = "Date"  # Set index name
            df.reset_index(inplace=True)  # Move date to a column

            # Save DataFrame as CSV
            filename = f"{self.output_dir}/{station}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")




object = StationDataProcessor(None, None)

object._process_file()

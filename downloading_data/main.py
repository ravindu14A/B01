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
        self.input_directory = "..\data"
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


        # Dictionary to store data for each station
        station_data = defaultdict(list)

        # Get all files in the input directory
        input_files = [f for f in os.listdir(self.input_directory) if
                       os.path.isfile(os.path.join(self.input_directory, f))]

        # Process each file
        station_dict = defaultdict(lambda: defaultdict(lambda: {"X": None, "Y": None, "Z": None}))

        for file_name in input_files:
            file_path = os.path.join(self.input_directory, file_name)
            self._process_file(file_path, station_data)

            info = _process_file(file_name)

            for name, pos in zip((info)[0], info[1]):
                in_dict = False
                for key in station_dict():
                    if name == key:
                        in_dict = True
                if in_dict:
                    station_dict[name][info[2]] = pos
                else:
                    station_dict[name] = {}
                    station_dict[name][info[2]] = pos

        return station_dict



    def _process_file(self, input_file):
        with open(input_file, "r") as file:
            context = file.readlines()

            match = re.search(r"(\d{2}[A-Z]{3}\d{2})", context[0])
            date = match.group(1)
            pos = []
            name = []
            counter = 0
            for line in context[1:]:
                split = line.split()
                if len(split) > 3:
                    if split[3] == "X":
                        X = split[4]
                    elif split[3] == "Y":
                        Y = split[4]
                    elif split[3] == "Z":
                        Z = split[4]
                    counter += 1
                    if counter > 2:
                        pos.append((X,Y,Z))
                        name.append(split[1])
                        counter = 0
            print(name)
            print(pos)

        return name, pos, date

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

object.process_files()





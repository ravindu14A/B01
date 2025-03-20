import os
from collections import defaultdict
from datetime import datetime
import re
import pandas as pd
import numpy as np
import Coordinate as geo

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

        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.station_dict = defaultdict(lambda: defaultdict(lambda: {"lat": None, "long": None, "alt": None}))

    def process_files(self):
        """
        Process all files in the input directory and create station-specific files.

        Returns:
            dict: A dictionary with station names as keys and output file paths as values
        """

        # Get all files in the input directory
        input_files = [f for f in os.listdir(self.input_directory) if
                       os.path.isfile(os.path.join(self.input_directory, f))]

        for file_name in input_files:
            file_path = os.path.join(self.input_directory, file_name)

            info = self._process_file(file_path)

            for name, pos, var in zip(info[0], info[1], info[3]):
                in_dict = False
                for key in self.station_dict.keys():
                    if name == key:
                        in_dict = True
                if in_dict:
                    self.station_dict[name][info[2]] = [pos, var]
                else:
                    self.station_dict[name] = {}
                    self.station_dict[name][info[2]] = [pos, var]


        self._write_station_files(self.station_dict)



    def _process_file(self, input_file):
        with open(input_file, "r") as file:
            context = file.readlines()

            match = re.search(r"(\d{2}[A-Z]{3}\d{2})", context[0])
            date = match.group(1)
            cov_dict = defaultdict(lambda: defaultdict(lambda: {}))
            matrix_XYZ = []  # cov-var matrix
            var_XYZ = []  # var
            cov_XYZ = []  # cov
            pos_XYZ = []
            pos_geo = []
            matrix_geo = []
            name = []
            counter = 0
            for line in context[1:]:
                split = line.split()
                if len(split) > 3:
                    if split[3] == "X":
                        X = split[4]
                        XX = split[6]
                    elif split[3] == "Y":
                        Y = split[4]
                        YY = split[6]
                    elif split[3] == "Z":
                        Z = split[4]
                        ZZ = split[6]
                    counter += 1
                    if counter > 2:
                        pos_XYZ.append((float(X), float(Y), float(Z)))
                        var_XYZ.append((float(XX), float(YY), float(ZZ)))
                        name.append(split[1])
                        counter = 0

                elif len(split) < 4:
                    in_dict = False
                    for key in cov_dict.keys():
                        if key == split[0]:
                            in_dict = True
                    if in_dict:
                        cov_dict[split[0]][split[1]] = split[2]
                    else:
                        cov_dict[split[0]] = {}
                        cov_dict[split[0]][split[1]] = split[2]

            for i in range(1, len(name) + 1):
                X = str(i * 3 - 2)
                Y = str(i * 3 - 1)
                Z = str(i * 3)

                if X in cov_dict and Y in cov_dict[X]:
                    xy = cov_dict[X][Y]
                else:
                    xy = cov_dict[Y][X]

                if X in cov_dict and Z in cov_dict[X]:
                    xz = cov_dict[X][Z]
                else:
                    xz = cov_dict[Z][X]

                if Y in cov_dict and Z in cov_dict[Y]:
                    yz = cov_dict[Y][Z]
                else:
                    yz = cov_dict[Z][Y]

                cov_XYZ.append((float(xy), float(xz), float(yz)))

            for var, cov in zip(var_XYZ, cov_XYZ):
                matrix = np.array([[var[0] ** 2, var[0]*var[1]*cov[0], var[0]*var[2]*cov[1]],
                                   [var[0]*var[1]*cov[0], var[1] ** 2, var[1]*var[2]*cov[2]],
                                   [var[0]*var[2]*cov[1], var[1]*var[2]*cov[2], var[2] ** 2]])
                matrix_XYZ.append(matrix)

            for pos, M in zip(pos_XYZ, matrix_XYZ):
                X = pos[0]
                Y = pos[1]
                Z = pos[2]
                position, var = geo.getGeodetic(X, Y, Z, M)
                pos_geo.append(position)
                matrix_geo.append(var)

        return name, pos_geo, date, matrix_geo

    def _write_station_files(self, station_data):
        """
        Saves station data as pickle files instead of CSV.

        Example:
            station_data = {
                'UHM3': {'2005': [1, 2, 3], '2006': [4, 2, 5]},
                'USHG': {'2003': [5, 6, 7], '2006': [6, 7, 8]},
            }
        """

        for station, data in station_data.items():
            # Transform dictionary into a DataFrame
            df = pd.DataFrame([
                [date, values[0][0], values[0][1], values[0][2], values[1]]  # Unpack tuple and keep array
                for date, values in data.items()
            ], columns=["Date", "lat", "long", "alt", "variance"])

            # Save DataFrame as a pickle file
            filename = os.path.join(self.output_directory, f"{station}.pkl")
            df.to_pickle(filename)

            # print(f"Saved: {filename}")


object = StationDataProcessor(r"..\..\data", r"..\processed_data\SE_Asia")

object.process_files()

# Load the DataFrame from the .pkl file
df = pd.read_pickle("../processed_data/SE_Asia/BABH.pkl")

pd.set_option("display.max_rows", None)   # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Prevent truncation of long values

# Display the DataFrame
print(df)














import os
from collections import defaultdict
import re
import pickle
import pandas as pd
import numpy as np
import preprocessing.coordinate as geo


# Load data
country = "malaysia"

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
            #add var, info[2]
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
#Covaraince part follows
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
        #add matrix_geo
        return name, pos_geo, date, matrix_geo

    def _write_station_files(self, station_data):
        """
        Saves station thailand as pickle files instead of CSV.

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
            ], columns=["date", "lat", "long", "alt", "covariance matrix"])
            df["date"] = pd.to_datetime(df["date"], format="%y%b%d")  # 💡 Notice the format change

            # Sort by Date (Earliest → Latest)
            df = df.sort_values(by="date").reset_index(drop=True)
            # Save DataFrame as a pickle file
            filename = os.path.join(self.output_directory, f"{station}.pkl")
            df.to_pickle(filename)

            # print(f"Saved: {filename}")


object = StationDataProcessor(f"../data/raw/{country}",
                              f"../data/partially_processed_steps/{country}/raw_pickle")

object.process_files()

###Filtering
directory = f"../data/partially_processed_steps/{country}/raw_pickle"
directory_out = f"../data/partially_processed_steps/{country}/filtered"


threshold= 300/7
cutoff = pd.Timestamp("2004-01-01 00:00:00")

for filename in os.listdir(directory_out):
    file_path = os.path.join(directory_out, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            # Do something with `thailand`
            df = pd.DataFrame(data)

            entries = len(df["date"])
            entry_time = df["date"][0]



            if entries>threshold and entry_time<cutoff:
                filename = os.path.join(directory_out, f"{filename}")
                df.to_pickle(filename)
                print(entries)


num_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
num_files_filtered = len([f for f in os.listdir(directory_out) if os.path.isfile(os.path.join(directory_out, f))])

print(f"""Number of files in original directory: {num_files}
Number of files in filtered directory: {num_files_filtered}""")














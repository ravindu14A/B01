import os
import pandas as pd

# Define the directory containing the files (current directory)
directory = '.'

# Initialize lists to store the results
avg_results = []
all_data_results = []
stations_with_20plus = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):  # Check if the file is a .pkl file
        file_path = os.path.join(directory, filename)

        # Read the data from the .pkl file
        try:
            data = pd.read_pickle(file_path)

            # Check if the required columns exist
            if all(col in data.columns for col in ['lat', 'long']):
                # Remove the .pkl extension from the file name
                file_name_without_extension = os.path.splitext(filename)[0]

                # Calculate the average latitude and longitude
                avg_latitude = data['lat'].mean()
                avg_longitude = data['long'].mean()

                # Get the number of data points (stations) in this file
                num_stations = len(data)

                # Append to averages results
                avg_results.append({
                    'File': file_name_without_extension,
                    'Average Latitude': avg_latitude,
                    'Average Longitude': avg_longitude,
                    'Number of samples': num_stations
                })

                # Append all raw data with file identifier
                data['Source File'] = file_name_without_extension
                all_data_results.append(data[['Source File', 'lat', 'long']])

                # If this file has more than 10 stations, add to special list
                if num_stations > 20:
                    stations_with_20plus.append({
                        'File': file_name_without_extension,
                        'Average Latitude': avg_latitude,
                        'Average Longitude': avg_longitude,
                        'Number of samples': num_stations
                    })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Convert the results to DataFrames
avg_df = pd.DataFrame(avg_results)
all_data_df = pd.concat(all_data_results, ignore_index=True)
stations_20plus_df = pd.DataFrame(stations_with_20plus)

# Save the results to CSV files
avg_df.to_csv('SE_Asia_avg_coordinates.csv', index=False)
all_data_df.to_csv('SE_Asia_coordinates.csv', index=False)
stations_20plus_df.to_csv('SE_Asia_stations_with_20plus.csv', index=False) # Rename for the number of

# Print the results
pd.options.display.max_rows = 9999
#print("Average Coordinates:")
#print(avg_df)
#print("\nStations with 20+ Data Points:")
print(stations_20plus_df)
print("\nAll Coordinates Sample:")
#print(all_data_df.head(999))
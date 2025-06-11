import os
import pandas as pd

# Define the directories containing the files
directories = [
    r'C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\Thailand\Raw_pickle',
    r'C:\Users\nicov\PycharmProjects\B01\preprocessing\processed_data\Malaysia\Raw_pickle'
]

# Initialize lists to store the results
avg_results = []
all_data_results = []
stations_with_104plus_samples = []

# Threshold for minimum number of data points
min_samples_threshold = 104

# Loop through both directories
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)

            try:
                data = pd.read_pickle(file_path)

                # Check if the required columns exist
                if all(col in data.columns for col in ['lat', 'long']):
                    # Remove the .pkl extension from the file name
                    station_name = os.path.splitext(filename)[0]

                    # Calculate average coordinates
                    avg_latitude = data['lat'].mean()
                    avg_longitude = data['long'].mean()

                    # Number of data points (samples)
                    num_samples = len(data)

                    # Append overall averages
                    avg_results.append({
                        'File': station_name,
                        'Average Latitude': avg_latitude,
                        'Average Longitude': avg_longitude,
                        'Number of Samples': num_samples
                    })

                    # Append raw data with file info
                    data['Source File'] = station_name
                    all_data_results.append(data[['Source File', 'lat', 'long']])

                    # Collect files with more than the minimum samples
                    if num_samples > min_samples_threshold:
                        stations_with_104plus_samples.append({
                            'File': station_name,
                            'Average Latitude': avg_latitude,
                            'Average Longitude': avg_longitude,
                            'Number of Samples': num_samples
                        })

            except Exception as e:
                print(f"Error reading {filename}: {e}")

# Convert the results to DataFrames
avg_df = pd.DataFrame(avg_results)
all_data_df = pd.concat(all_data_results, ignore_index=True)
stations_104plus_samples_df = pd.DataFrame(stations_with_104plus_samples)

# Save the results to CSV files
avg_df.to_csv('SE_Asia_avg_coordinates.csv', index=False)
all_data_df.to_csv('SE_Asia_coordinates.csv', index=False)
stations_104plus_samples_df.to_csv('SE_Asia_files_with_104plus_samples.csv', index=False)

# Print the results
pd.options.display.max_rows = 9999
print("\nFiles with more than 104 data points:")
print(stations_104plus_samples_df)

print("\nAll Coordinates Sample:")
print(all_data_df.head(99999))

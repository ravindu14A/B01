import math as mt
import pandas as pd
import matplotlib.pyplot as plt


magnitude_selector = 8 # The lowest magnitude of earthquakes to be processed from the csv files below

# Load the CSV file
df = pd.read_csv(r"C:\Users\nicov\Downloads\SE_asia_Earthquake_history.csv")
all_earthquakes = pd.read_csv(r"C:\Users\nicov\Downloads\query.csv")

# Extract relevant columns
latitude = all_earthquakes['latitude']
longitude = all_earthquakes['longitude']
magnitude = all_earthquakes['mag']
time_frame = all_earthquakes['time']

#  Use pandas to filter the data
filtered_data = all_earthquakes[all_earthquakes['mag'] > magnitude_selector]

# Create a list of tuples (lat, lon, mag, time) for filtered data
coords_try = list(zip(filtered_data['latitude'], filtered_data['longitude'], filtered_data['mag'], filtered_data['time']))

print(coords_try)  # Check the filtered dataset
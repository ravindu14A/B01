import DataFrame_genrator
df = DataFrame_genrator.GetSavedData()


import pandas as pd
import matplotlib.pyplot as plt

# Ensure Date is in datetime format and sort the data by time
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Prompt for station name
station_name = input("Enter the station name: ")

# Filter for the desired station
df_station = df[df["Station"] == station_name]

# Set up the figure with 3 subplots (one for each coordinate)
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot Latitude
axs[0].plot(df_station["Date"], df_station["Latitude"], marker="o", linestyle="-", markersize=2, color='b')
axs[0].set_title(f"Latitude Over Time for Station {station_name}")
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Latitude")
axs[0].grid(True, linestyle="--", alpha=0.6)

# Plot Longitude
axs[1].plot(df_station["Date"], df_station["Longitude"], marker="s", linestyle="-", markersize=2, color='g')
axs[1].set_title(f"Longitude Over Time for Station {station_name}")
axs[1].set_xlabel("Date")
axs[1].set_ylabel("Longitude")
axs[1].grid(True, linestyle="--", alpha=0.6)

# Plot Height
axs[2].plot(df_station["Date"], df_station["Height"], marker="^", linestyle="-", markersize=2, color='r')
axs[2].set_title(f"Height Over Time for Station {station_name}")
axs[2].set_xlabel("Date")
axs[2].set_ylabel("Height")
axs[2].grid(True, linestyle="--", alpha=0.6)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


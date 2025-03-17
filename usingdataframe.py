import pandas as pd
import numpy as np


df = pd.read_csv("output1.csv")
station_name = "ARAU"
filtered_df = df[df["Station"] == station_name]

#print(interested)
print(filtered_df[:3])
import matplotlib.pyplot as plt
import pandas as pd

# Ensure 'Date' column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])  

# Filter data for a specific station
station_name = "AATSR"  # Change as needed
filtered_df = df[df["Station"] == station_name].sort_values("Date")  # Sort by date

# Extract X positions properly
times = filtered_df["Date"]

# Convert "Position" column to NumPy arrays and extract X values
x_positions = filtered_df["Position"].apply(lambda pos: pos[0] if isinstance(pos, (list, tuple, np.ndarray)) else float(pos.strip("[]").split(",")[0]))


print(x_positions[:3])
# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(times, x_positions, color='b', label=f"X Position ({station_name})", alpha=0.7)

# Formatting
plt.xlabel("Time")
plt.ylabel("X Position")
plt.title(f"X Position vs. Time for {station_name}")
plt.xticks(rotation=45)  # Rotate for better readability
plt.grid(True)
plt.legend()
plt.show()

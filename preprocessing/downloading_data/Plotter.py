import pandas as pd
import matplotlib.pyplot as plt
import main as mn

# Load data
country = mn.country
station = mn.station

# Load the DataFrame from the .pkl file
df = pd.read_pickle(f"../processed_data/{country}/Final/{station}.pkl")
# Show all rows
pd.set_option('display.max_rows', None)

# Show all columns
pd.set_option('display.max_columns', None)

# Optional: prevent column width truncation
pd.set_option('display.max_colwidth', None)  # or use pd.set_option('display.max_colwidth', -1) in older versions

# Optional: expand frame to full width of console
pd.set_option('display.width', None)
print(df["date"])
print(df["lat"])
pd.set_option("display.max_rows", None)   # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Prevent truncation of long values

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)  # 3 subplots stacked

# Plot Latitude vs. Date
axes[0].plot(df["date"], df["lat"], marker="o", linestyle="-", markersize=1, label="Latitude", color = "blue")
axes[0].set_ylabel("North_South (cm) / Lat")
axes[0].legend()
axes[0].grid(True)
axes[0].set_title(f"{station} Data Over Time", fontsize=14, fontweight="bold")

# Plot Longitude vs. Date
axes[1].plot(df["date"], df["long"], marker="o", linestyle="-", markersize=1, label="Longitude", color="green")
axes[1].set_ylabel("East_West (cm) / Long")
axes[1].legend()
axes[1].grid(True)

# Plot Altitude vs. Date
# axes[2].plot(df["date"], df["alt"], marker="o", linestyle="-", markersize=1, label="Altitude", color="red")
# axes[2].set_ylabel("Altitude (abs in m)")
# axes[2].set_xlabel("Date")  # Only the last subplot needs x-axis label
# axes[2].legend()
# axes[2].grid(True)


# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust spacing
plt.tight_layout()
# plt.savefig("normalised.png")
# plt.savefig("original.png")
# Show the figure
plt.show()

import numpy as np

# Define matrix A and vector b
A = np.array([[1, 1,1,1,1], [0, -1,1,-1,1], [0,-1,1,1,-1], [0, 0.5, 0.5, 0.5,0.5], [0,1,1,-1,-1]])
b = np.array([0,0,0,0,1])

# Solve for x
x = np.linalg.solve(A, b)

print("Solution x:", x)
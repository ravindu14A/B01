import pandas as pd
import matplotlib.pyplot as plt

country = "Thailand"
station = "PHUK"

# Load the DataFrame
df = pd.read_pickle(f"../processed_data/{country}/Filtered_cm/{station}.pkl")
df1 =pd.read_pickle(f"../processed_data/{country}/Filtered_cm_normalised/{station}.pkl")
# Display settings (optional)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Create 2 vertically stacked subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

# Plot Latitude vs. Date
axes[0,0].plot(df["date"], df["lat"], marker="o", linestyle="-", markersize=1, color="blue", label="Latitude")
axes[0,0].set_ylabel("North_South (cm)")
axes[0,0].set_title(f"{station} Absolute Plate Motion NOT Removed", fontsize=10)
axes[0,0].legend()
axes[0,0].grid(True)

# Plot Longitude vs. Date
axes[1,0].plot(df["date"], df["long"], marker="o", linestyle="-", markersize=1, color="green", label="Longitude")
axes[1,0].set_ylabel("East_West (cm)")
axes[1,0].set_xlabel("Date")
axes[1,0].legend()
axes[1,0].grid(True)

# Plot Longitude vs. Date
axes[0,1].plot(df["date"], df1["lat"], marker="o", linestyle="-", markersize=1, color="blue", label="Latitude")
axes[0,1].set_ylabel("North_South (cm)")
axes[0,1].set_title(f"{station} Absolute Plate Motion Removed", fontsize=10)
axes[0,1].legend()
axes[0,1].grid(True)

# Plot Longitude vs. Date
axes[1,1].plot(df["date"], df1["long"], marker="o", linestyle="-", markersize=1, color="green", label="Longitude")
axes[1,1].set_ylabel("East_West (cm)")
axes[1,1].set_xlabel("Date")
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

country = "Malaysia"
station = "ARAU"

# Load the DataFrame
df = pd.read_pickle(f"../processed_data/{country}/Filtered_cm/{station}.pkl")

# Display settings (optional)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Create 2 vertically stacked subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot Latitude vs. Date
axes[0].plot(df["date"], df["lat"], marker="o", linestyle="-", markersize=1, color="blue", label="Latitude")
axes[0].set_ylabel("North_South (cm)")
axes[0].set_title(f"{station} Latitude and Longitude Over Time", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].grid(True)

# Plot Longitude vs. Date
axes[1].plot(df["date"], df["long"], marker="o", linestyle="-", markersize=1, color="green", label="Longitude")
axes[1].set_ylabel("East_West (cm)")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

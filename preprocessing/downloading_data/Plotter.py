import pandas as pd
import matplotlib.pyplot as plt

station = ("CBAS")

# Load the DataFrame from the .pkl file
df = pd.read_pickle(f"../processed_data/SE_Asia/{station}.pkl")

pd.set_option("display.max_rows", None)   # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Prevent truncation of long values

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)  # 3 subplots stacked

# Plot Latitude vs. Date

axes[0].plot(df["Date"], df["lat"], marker="o", linestyle="-", markersize=2, label="Latitude", color = "blue")
axes[0].set_ylabel("Latitude")
axes[0].legend()
axes[0].grid(True)
axes[0].set_title(f"{station} Data Over Time", fontsize=14, fontweight="bold")

# Plot Longitude vs. Date
axes[1].plot(df["Date"], df["long"], marker="o", linestyle="-", markersize=2, label="Longitude", color="green")
axes[1].set_ylabel("Longitude")
axes[1].legend()
axes[1].grid(True)

# Plot Altitude vs. Date
axes[2].plot(df["Date"], df["alt"], marker="o", linestyle="-", markersize=2, label="Altitude", color="red")
axes[2].set_ylabel("Altitude")
axes[2].set_xlabel("Date")  # Only the last subplot needs x-axis label
axes[2].legend()
axes[2].grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust spacing
plt.tight_layout()
# Show the figure
plt.show()

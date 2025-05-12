import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, bisect

# --- Load data ---
station = "PHUK"
country = "Thailand"
filepath = f"../processed_data/{country}/Final/{station}.pkl"

with open(filepath, 'rb') as f:
    data = pickle.load(f)
    df = pd.DataFrame(data)

# --- Preprocessing ---
df['date'] = pd.to_datetime(df['date'])
start_date = df["date"].iloc[0]
end_date = pd.to_datetime("2004-12-15")
df['days_from_ref'] = (df['date'] - start_date).dt.days

# --- First segment for linear fit ---
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
df_subset = df.loc[mask]
X = df_subset['days_from_ref'].values.reshape(-1, 1)
y = df_subset['lat'].values

# --- Fit linear regression ---
model = LinearRegression()
model.fit(X, y)
slope_per_day = model.coef_[0]
slope_mm_per_year = slope_per_day * 10 * 365
print(f"Slope (mm/year): {slope_mm_per_year:.3f}")

y_pred = model.predict(X)

# --- Fit exponential recovery model ---
df_fit = df.loc[df['date'] > end_date]
start_day1 = df_fit["days_from_ref"].iloc[0]
t_data = df_fit['days_from_ref'].values
y_data = df_fit['lat'].values

# Model definition
def model_func(t, A, B, c1, c2, d):
    return A * np.exp(-c1 * (t - start_day1)) + B * np.exp(-c2 * (t - start_day1)) + d + slope_per_day * (t - start_day1)

# Fit the model
popt, pcov = curve_fit(model_func, t_data, y_data)
A, B, c1, c2, d = popt
param_names = ['A', 'B', 'c1', 'c2', 'd']
for name, val in zip(param_names, popt):
    print(f"{name} = {val:.6f}")

# Generate future prediction points
years = 340
safe_end = round(years* 365.25)  # ~547 years
T = np.arange(start_day1, safe_end, 5)
y_fit = model_func(T, *popt)
ref = np.full_like(T, df_fit["lat"].iloc[0])

# Root finding
def f(t):
    return model_func(t, *popt) - df_fit["lat"].iloc[0]

plt.figure(figsize=(12, 6))

try:
    root = bisect(f, 10000, safe_end - 1)
    y_point = model_func(root, *popt)
    root_years = root / 365.25  # ✅ Use absolute time axis, not offset

    # Plot predicted event point
    plt.scatter(root_years, y_point, color='blue', zorder=5)
    plt.text(root_years + 5, y_point - 2, f"Predicted EQ\n~Year {int(start_date.year + root / 365.25)}", fontsize=10)
    print(f"Earthquake prediction: {root_years} years after {start_date}",
          f"(~Year {int(start_date.year + root / 365.25)})")

except Exception as e:
    print("Adjust bisection settings:", e)

# --- Plot ---
T_years = T / 365.25  # ✅ Correct x-axis alignment
df_years = df['days_from_ref'] / 365.25
df_subset_years = df_subset['days_from_ref'] / 365.25



plt.plot(T_years, y_fit, 'r-', label='Fitted Curve')
plt.plot(df_years, df['lat'], label='Original Data', color='green')
plt.plot(df_subset_years, y_pred, color='red', label='Linear Fit Segment')
plt.plot(T_years, ref, color='black', linestyle="--", label='Initial Lat')

plt.xlabel('Years since reference date')
plt.ylabel('Latitude (PCA transformed)')
plt.title(f'{station} - Earthquake Prediction (Relative Time)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

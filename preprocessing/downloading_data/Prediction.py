import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, bisect
import pickle
import pandas as pd
import matplotlib.pyplot as plt
# import main as mn

# Load data
country = "Malaysia"
station = "IPOH"
filepath = f"../processed_data/{country}/Final/{station}.pkl"
years = 1000

with open(filepath, 'rb') as f:
    data = pickle.load(f)
    df = pd.DataFrame(data)

df['date'] = pd.to_datetime(df['date'])
start_date = df["date"].iloc[0]
end_date = pd.to_datetime("2004-12-15")
df['days_from_ref'] = (df['date'] - start_date).dt.days

# === Linear Fit Before Earthquake ===
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
df_subset = df.loc[mask]
X = df_subset['days_from_ref'].values.reshape(-1, 1)
y = df_subset['lat'].values
print(df)
model = LinearRegression()
model.fit(X, y)
slope_per_day = model.coef_[0]
slope_mm_per_year = slope_per_day * 10 * 365
print(f"Linear velocity (mm/year): {slope_mm_per_year:.4f}")

y_pred = model.predict(X)

# === Post-Earthquake Fit ===
df_fit = df.loc[df['date'] > end_date]
start_day1 = df_fit["days_from_ref"].iloc[0]
t_data = df_fit['days_from_ref'].values
y_data = df_fit['lat'].values

def model_func(t,A, B,c1, c2, d):
    return A * np.exp(-c1 * (t - start_day1)) + B * np.exp(-c2 * (t - start_day1)) + d + slope_per_day * (t - start_day1)

popt, pcov = curve_fit(model_func, t_data, y_data, maxfev = 10000)

param_names = ['A', 'B', 'c1', 'c2', 'd']
print("\nCurve fit parameters:")
for name, val in zip(param_names, popt):
    print(f"{name} = {val:.6f}")

# === Prediction & Intersection ===

safe_end = round(years * 365.25)
T = np.arange(start_day1, safe_end, 5)
y_fit = model_func(T, *popt)


def f(t):
    return model_func(t, *popt) - df_fit["lat"].iloc[0]

# === Plotting ===
plt.figure(figsize=(12, 6))

try:
    root = bisect(f, start_day1+2000, safe_end - 1)
    root_years = root / 365.25
    y_point = model_func(root, *popt)

    # Mark intersection
    plt.scatter(root_years, y_point, color='blue', zorder=5, label="Predicted Intersection")
    plt.axhline(df_fit["lat"].iloc[0], color='black', linestyle="--", linewidth=1, label='Initial Lat')
    plt.text(root_years + 2, y_point -8,
             f"Predicted EQ\n~Year {int(start_date.year + root / 365.25)}",
             fontsize=10, color='blue')

    print(f"\nEarthquake prediction: {root_years:.2f} years after {start_date} "
          f"(~Year {int(start_date.year + root / 365.25)})")

except Exception as e:
    print("Adjust bisection settings:", e)


# === Plot All Data and Fits ===
T_years = T / 365.25
df_years = df['days_from_ref'] / 365.25
df_subset_years = df_subset['days_from_ref'] / 365.25


plt.plot(T_years, y_fit, 'r-', label='Fitted Curve')
plt.plot(df_years, df['lat'], label='Original Data', color='green')
plt.plot(df_subset_years, y_pred, color='red', label='Linear Fit Segment')

plt.xlabel('Years since reference date')
plt.ylabel('Position (PCA transformed)')
plt.title(f'{station} - Earthquake Prediction since {start_date}')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()




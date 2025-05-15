import numpy as np
import os
import pickle
import pandas as pd
import PCA as pc
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, bisect
import main as mn

# Load data
country = mn.country
station = mn.station
directory_out = f"../processed_data/{country}/Error"
filepath = f"../processed_data/{country}/Raw_pickle/{station}.pkl"
N = 100
n_tsd = 2.4477
confidence = 1.96 #95% confidence

with open(filepath, 'rb') as f:
    data = pickle.load(f)
    df = pd.DataFrame(data)

# Process covariance matrices
covariance_msqr = df["covariance matrix"]
covariance_cmsqr = covariance_msqr.apply(lambda mat: mat * 1e4)
covariance_2d = covariance_cmsqr.apply(lambda M: M[:2, :2])

# Apply PCA transformation
P = pc.trans[f"{station}"]
covariance_pca_2d = covariance_2d.apply(lambda M: P @ M @ P.T)

# Load PCA coordinates
df1 = pd.read_pickle(f"../processed_data/{country}/Final/{station}.pkl")
north = df1["lat"].to_numpy()
east = df1["long"].to_numpy()
position = df1["lat"].to_numpy()


# Error ellipse plotting function
def plot_error_ellipse(cov, mean, ax = None, n_std=1.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    # ax.add_patch(ellipse)
    std = np.sqrt(vals[0])
    return n_tsd* std

# Create figure with correct size
# fig, ax = plt.subplots(figsize=(10, 10))  # Adjust as needed
standard_deviation = []
# Plot ellipses and points
for n, e, cov in zip(north, east, covariance_pca_2d):
    mean = [n, e]
    std = plot_error_ellipse(cov, mean, n_std=2, edgecolor='red', facecolor='none')
    standard_deviation.append(std)


# Plot all center points
# ax.scatter(north, east, color='blue', label='Center', s=10)
# standard_deviation = np.array(standard_deviation)
# # Format plot
# ax.set_aspect('equal')
# ax.set_xlabel('PCA 1 (cm)')
# ax.set_ylabel('PCA 2 (cm)')
# ax.set_title('2σ Error Ellipse in PCA Space')
# ax.legend()
# ax.grid(True)

# plt.show()

all_samples = []
np.random.seed(3)

for i in range(N):
    y_sampled = np.random.normal(loc=position, scale=standard_deviation)
    # e.g., fit a curve here and store the result
    all_samples.append(y_sampled)


# fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)  # 3 subplots stacked

# Plot Latitude vs. Date
# axes[0].plot(df["date"],all_samples[0], marker="o", linestyle="-", markersize=1, label="Latitude", color = "blue")
# axes[0].set_ylabel("North_South (cm) / Lat")
# axes[0].legend()
# axes[0].grid(True)
# axes[0].set_title(f"{station} Data Over Time", fontsize=14, fontweight="bold")

# plt.show()




results = []
for i in range(N):
    df2 = pd.DataFrame()
    df2["date"] = df['date']
    df2["lat"] = all_samples[i]


    df2['date'] = pd.to_datetime(df2['date'])
    start_date = df2["date"].iloc[0]
    end_date = pd.to_datetime("2004-12-15")
    df2['days_from_ref'] = (df2['date'] - start_date).dt.days

    # === Linear Fit Before Earthquake ===
    mask = (df2['date'] >= start_date) & (df2['date'] <= end_date)
    df_subset = df2.loc[mask]
    X = df_subset['days_from_ref'].values.reshape(-1, 1)
    y = df_subset['lat'].values

    model = LinearRegression()
    model.fit(X, y)
    slope_per_day = model.coef_[0]
    slope_mm_per_year = slope_per_day * 10 * 365
    # print(f"Slope (cm/year): {slope_per_day}")

    y_pred = model.predict(X)

    # === Post-Earthquake Fit ===
    df_fit = df2.loc[df['date'] > end_date]
    start_day1 = df_fit["days_from_ref"].iloc[0]
    t_data = df_fit['days_from_ref'].values
    y_data = df_fit['lat'].values

    def model_func(t, A, B, c1, c2, d):
        return A * np.exp(-c1 * (t - start_day1)) + B * np.exp(-c2 * (t - start_day1)) + d + slope_per_day * (t - start_day1)

    popt, pcov = curve_fit(model_func, t_data, y_data)
    #A, B, c1, c2, d = popt


    #param_names = ['A', 'B', 'c1', 'c2', 'd']
    # print("\nCurve fit parameters:")
    # for name, val in zip(param_names, popt):
    #     print(f"{name} = {val:.6f}")


    years = 400
    safe_end = round(years * 365.25)
    T = np.arange(start_day1, safe_end, 5)
    y_fit = model_func(T, *popt)

    def f(t):

        return model_func(t, *popt) - df_fit["lat"].iloc[0]

    # === Plotting ===
    # plt.figure(figsize=(12, 6))

    try:
        root = bisect(f, start_day1+10000, safe_end)
        root_years = root / 365.25
        y_point = model_func(root, *popt)
        #
        # Mark intersection
        # plt.scatter(root_years, y_point, color='blue', zorder=5, label="Predicted Intersection")
        # plt.axhline(df_fit["lat"].iloc[0], color='black', linestyle="--", linewidth=1, label='Initial Lat')
        # plt.text(root_years + 2, y_point -8,
        #          f"Predicted EQ\n~Year {int(start_date.year + root / 365.25)}",
        #          fontsize=10, color='blue')

        # print(f"\nEarthquake prediction: {root_years:.2f} years after {start_date} "
        #       f"(~Year {int(start_date.year + root / 365.25)})")

        results.append((y_pred, slope_per_day, root_years, popt))

    except Exception as e:
        print("Adjust bisection settings:", e)
        plt.plot(T ,f(T), label='Original Data', color='green')
        plt.show()


    # === Plot All Data and Fits ===
    T_years = T / 365.25
    df_years = df2['days_from_ref'] / 365.25
    df_subset_years = df_subset['days_from_ref'] / 365.25

    # plt.plot(T_years, y_fit, 'r-', label='Fitted Curve')
    # plt.plot(df_years, df2['lat'], label='Original Data', color='green')
    # plt.plot(df_subset_years, y_pred, color='red', label='Linear Fit Segment')
    #
    # plt.xlabel('Years since reference date')
    # plt.ylabel('Position (PCA transformed)')
    # plt.title(f'{station} - Earthquake Prediction (Relative Time)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

y_preds, slope, predictions, parameters = zip(*results)
y_preds = list(y_preds)
slope = list(slope)
predictions = list(predictions)
parameters = list(parameters)
values = predictions
mean = np.mean(values)
variance = np.var(values)
std_dev = np.std(values)
minimum = np.min(values)
maximum = np.max(values)
median = np.median(values)

print(f"Earthquake prediction: {mean} +/- {confidence*std_dev} years afer {start_date}")


def model_func(v, t, A, B, c1, c2, d):
    return A * np.exp(-c1 * (t - start_day1)) + B * np.exp(-c2 * (t - start_day1)) + d + v * (
                t - start_day1)

def find_closest(nums, target):
    closest_index = min(range(len(nums)), key=lambda i: abs(nums[i] - target))
    return nums[closest_index], closest_index

lower = find_closest(predictions, mean - std_dev*confidence)
l = lower[0]
l_params = parameters[lower[1]]
v_l =slope[lower[1]]
y_lfit = model_func(v_l, T, *l_params)


upper = find_closest(predictions, mean + std_dev*confidence)
u = upper[0]
u_params = parameters[upper[1]]
v_u =slope[upper[1]]
y_ufit = model_func(v_u, T, *u_params)

mn = find_closest(predictions, mean)
m = mn[0]
m_params = parameters[mn[1]]
v_m =slope[mn[1]]
y_mfit = model_func(v_m, T, *m_params)

y_pred1= y_preds[lower[1]]
y_pred2=y_preds[upper[1]]

plt.figure(figsize=(15, 7))
x_err = np.array([[u-m],[m-l]])
plt.errorbar(predictions[mn[1]], df_fit["lat"].iloc[0], xerr= x_err, fmt=' ', ecolor='black', capsize=5)
plt.text(root_years-40, y_point+5,
         f"Uncertainty: +/- {np.round(std_dev*confidence, 1)}",
         fontsize=8, color='black')
plt.scatter(predictions[mn[1]],df_fit["lat"].iloc[0] , color='blue', zorder=5, label="Predicted Intersection")
plt.axhline(df_fit["lat"].iloc[0], color='black', linestyle="--", linewidth=1, label='Initial Position')
plt.text(root_years-5, y_point -12,
         f"Predicted EQ\n~Year {int(start_date.year + root / 365.25)}",
         fontsize=10, color='blue')
plt.fill_between(T_years, y_lfit, y_ufit, color='red', alpha=0.3, label='95% CI')
plt.plot(T_years, y_mfit, 'b-', label='Mean')
plt.plot(df_years, df2['lat'], label='Original Data', color='green')


plt.xlabel('Years since reference date')
plt.ylabel('PCA Transformed Position (cm)')
plt.title(f'{station} - Earthquake Prediction - Reference: {start_date.date()}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Prediction.png")
plt.show()

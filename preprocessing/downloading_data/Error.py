import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, bisect
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PCA import trans
# import main as mn


# Load data
country = "Thailand"
station = "PHUK"
N = 1000
bar = 115
follow = 0
years = 350
std_n = 1.96
filepath = f"../processed_data/{country}/Filtered_cm/{station}.pkl"
with open(filepath, 'rb') as f:
    data = pickle.load(f)
    df = pd.DataFrame(data)
    tod_covariance = df["covariance matrix"].apply(lambda m: m[:2, :2])
    tod_covariance = tod_covariance.rename("covariance matrix").to_frame()
    tod_covariance["covariance matrix"] = tod_covariance["covariance matrix"].apply(lambda M: 10000 * M)

    V = trans[station]
    tod_covariance_pca = tod_covariance["covariance matrix"].apply(lambda S: V.T @ S @ V)


#standard deviation
standard_deviations = []
for A in tod_covariance_pca:
    eigenvalues, _ = np.linalg.eig(A)

    # Get the largest eigenvalue
    largest = np.max(eigenvalues)



    standard_deviations.append(np.sqrt(largest))


standard_deviations = pd.DataFrame({"standard deviation": standard_deviations})



filepath = f"../processed_data/{country}/Final/{station}.pkl"
with open(filepath, 'rb') as f:
    data = pickle.load(f)
    df = pd.DataFrame(data)
    start_date = df["date"].iloc[0]
    end_date = pd.to_datetime("2004-10-01")


    df1 = pd.DataFrame(data)
    df1['date'] = pd.to_datetime(df['date'])
    mask = (df1['date'] >= start_date) & (df1['date'] <= end_date)
    df1_subset = df1.loc[mask]
    df1_fit = df1.loc[df['date'] > end_date]



predictions = []
parameters = []
slope = []
np.random.seed(2)
for i in range(N):
    samples = np.random.normal(loc=df1["lat"], scale=standard_deviations["standard deviation"])

    # Save to new column in df1
    df["lat"] = samples
    df['date'] = pd.to_datetime(df['date'])
    df['days_from_ref'] = (df['date'] - start_date).dt.days

    # === Linear Fit Before Earthquake ===
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_subset = df.loc[mask]
    X = df_subset['days_from_ref'].values.reshape(-1, 1)
    y = df_subset['lat'].values



    model = LinearRegression()
    model.fit(X, y)
    slope_per_day = model.coef_[0] #cm/day
    slope_mm_per_year = slope_per_day * 10 * 365


    y_pred = model.predict(X)

    # === Post-Earthquake Fit ===
    df_fit = df.loc[df['date'] > end_date]
    start_day1 = df_fit["days_from_ref"].iloc[0]
    t_data = df_fit['days_from_ref'].values
    y_data = df_fit['lat'].values

    def model_func(t,A, B,c1, c2, d):
        return A * np.exp(-c1 * (t - start_day1)) + B * np.exp(-c2 * (t - start_day1)) + d + slope_per_day * (t - start_day1)

    popt, pcov = curve_fit(model_func, t_data, y_data, maxfev = 10000)


    # === Prediction & Intersection ===

    safe_end = round(years * 365.25)
    T = np.arange(start_day1, safe_end, 5)
    y_fit = model_func(T, *popt)


    def f(t):
        return model_func(t, *popt) - df_fit["lat"].iloc[0]





    try:
        root = bisect(f, start_day1+2000, safe_end - 1)
        root_years = root / 365.25
        y_point = model_func(root, *popt)
        predictions.append(root_years)
        parameters.append(popt)
        slope.append(slope_per_day)

    except Exception as e:
        print("Adjust bisection settings:", e)


def find_closest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))

mean = np.mean(predictions)
std = np.std(predictions)  # By default, uses population std (ddof=0)
lower = mean - std_n * std
upper = mean + std_n * std

index_lower = find_closest_index(predictions, lower)
index_mean = find_closest_index(predictions, mean)
index_upper = find_closest_index(predictions, upper)

lower_params = parameters[index_lower]
mean_params = parameters[index_mean]
upper_params = parameters[index_upper]

lower_v = slope[index_lower]
mean_v = slope[index_mean]
upper_v = slope[index_upper]



##====== Func for plottinf error
def model_func(t, v,  A, B, c1, c2, d):
    return A * np.exp(-c1 * (t - start_day1)) + B * np.exp(-c2 * (t - start_day1)) + d + v * (t - start_day1)

y_lower = model_func(T, lower_v, *lower_params)
y_mean = model_func(T, mean_v, *mean_params)
y_upper = model_func(T, upper_v, *upper_params)


# === Plotting ===
# === Plot All Data and Fits ===
T_years = T / 365.25
df_years = df['days_from_ref'] / 365.25
df_subset_years = df_subset['days_from_ref'] / 365.25



plt.figure(figsize=(7, 4.5))


#Mark intersection
plt.errorbar(
    mean, df1_fit["lat"].iloc[0],
    xerr=[[mean-lower], [upper-mean]],  # asymmetric horizontal errors
    fmt='o',  # or '' for no point
    ecolor='black',
    capsize=5
)


plt.scatter(mean, df1_fit["lat"].iloc[0], color='blue', zorder=5, label="Predicted Intersection")
plt.axhline(df1_fit["lat"].iloc[0], color='black', linestyle="--", linewidth=1, label='Initial Position')
plt.text(years - bar, df1_fit["lat"].iloc[0] + 1,
         f"Predicted EQ\n~Year {int(start_date.year + mean)}",
         fontsize=8, color='blue')
plt.text(root_years - bar - follow, df1_fit["lat"].iloc[0] + 1,
         f"Uncertainty: +/- {round(std_n * std,1)}",
         fontsize=8, color='black')


print(f"\nEarthquake prediction: {root_years:.2f} years after {start_date} "
      f"(~Year {int(start_date.year + root / 365.25)})")



plt.fill_between(T_years, y_lower, y_upper, color='red', alpha=0.3, label='Confidence Interval')
plt.plot(T_years, y_mean, 'b-', label='Mean')
plt.plot(df_subset_years, y_pred, color='red', label='Linear Fit Segment')
plt.plot(df_years, df1['lat'], label='Processed Data', color='green')

plt.xlabel('Years since reference date')
plt.ylabel('PCA Transformed Position (cm)')
plt.title(f'{station} - Earthquake Prediction | Reference Date: {start_date.date()}')
plt.legend(
    fontsize=8,           # Adjust legend text size
    loc='lower right',    # Bottom-right corner
    frameon=True,         # Optional: box around legend
    borderpad=0.5,        # Padding inside the box
    labelspacing=0.4      # Space between legend entries
)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Prediction_{country}_{station}")
plt.show()




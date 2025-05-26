import pickle
import pandas as pd
from PCA import yohooo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Replace 'your_file.pkl' with the actual file path
with open(r'../../data/partially_processed/Thailand/PCA/PHUK.pkl', 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the object that was saved in the pickle file
with open(r'../../data/partially_processed/Thailand/Filtered_cm/PHUK.pkl', 'rb') as file:
    cov_ARAU = pickle.load(file)

merged_df = pd.merge(data, cov_ARAU[['date', 'covariance matrix']], on='date', how='inner')  # Options: 'inner', 'outer', 'left', 'right'

def new_covariance(matrix):
    """
    Args:
        matrix: 3x3 matrix north, east, up

    Returns: single value of interest
    """
    slic_matrix = matrix[:2, :2]
    pca = yohooo()
    pca_var = pca.components_[0]
    return pca_var @ slic_matrix @ pca_var.T

def smart_epsilon(x0):
    eps_machine = np.finfo(float).eps
    return np.sqrt(eps_machine) * np.maximum(np.abs(x0), 1.0)
print(merged_df.iloc[0]['covariance matrix'])
merged_df['covariance matrix'] = merged_df.apply(lambda row:new_covariance(row['covariance matrix']), axis=1)

# Convert date to datetime just to be sure
merged_df['date'] = pd.to_datetime(merged_df['date'])

# Set reference and end date (same logic as in second file)
start_date = merged_df['date'].iloc[0]
end_date = pd.to_datetime("2004-12-15")

# Add numeric time axis
merged_df['days_from_ref'] = (merged_df['date'] - start_date).dt.days

# Subset the data for slope estimation
df_subset = merged_df[merged_df['date'] <= end_date]

# Prepare features and target
X = df_subset['days_from_ref'].values.reshape(-1, 1)
y = df_subset['lat'].values

# Fit linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
slope_per_day = model.coef_[0]

# Optional print
slope_mm_per_year = slope_per_day * 10 * 365
print(f"Slope (mm/year): {slope_mm_per_year:.3f}")



merged_df = merged_df.iloc[350:]



print(merged_df)

# List of diagonal values
diag_vals = merged_df['covariance matrix'].values.tolist()

# Create diagonal matrix
D = np.diag(diag_vals)
D = D * 10000

y = merged_df['lat'].values

# y: known data points (shape: T)
# cov: known covariance matrix of y (shape: T x T)
N = 50  # number of Monte Carlo samples

simulated_datasets = np.random.multivariate_normal(y, D, size=N)

#--------------------------------------------
plt.figure(figsize=(12, 6))

for i in range(N):
    # Step 1: Extract date and value
    dates = merged_df['date'].to_list()
    values = merged_df['lat'].to_numpy()


    # Step 2: Convert dates to numeric (e.g., days since first date)
    base_date = dates[0]
    t_numeric = np.array([(d - base_date).days / 365.0 for d in dates])  # Convert days to years

    # Step 3: Fit quadratic function
    def model_func(t, A, B, c1, c2, d):
        return A * np.exp(-c1 * (t - t_numeric[0])) + B * np.exp(-c2 * (t - t_numeric[0])) + d + slope_per_day * (
                    t - t_numeric[0])


    params, _ = curve_fit(model_func, t_numeric, simulated_datasets[i])

    # Step 4: Create future dates (e.g., 30 years = 30*365 days)
    future_days = 400 * 365

    t_future_numeric = np.arange(t_numeric[-1] + 1, t_numeric[-1] + future_days + 1) / 365.0  # Future in years
    #future_dates = [base_date + pd.Timedelta(days=int(d)) for d in t_future_numeric]

    # Step 5: Predict future values
    future_preds = model_func(t_future_numeric, *params)
    plt.plot(t_future_numeric, future_preds, label='30-Year Forecast', color='red')


# Step 6: Plot original and predicted
plt.plot(t_numeric, values, label='Observed', color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Quadratic Fit and 30-Year Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

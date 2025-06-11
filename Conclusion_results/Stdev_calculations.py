import numpy as np
from scipy.stats import norm

# ---- Data from LaTeX table ----
predictions = [
    ("ARAU",     "Malaysia", 2331, 24.2),
    ("GETI",     "Malaysia", 2190, 15.2),
    ("KUAL",     "Malaysia", 2382, 53.6),
    ("USMP",     "Malaysia", 2182, 6.6),
    ("PHUK",     "Thailand", 2290, 10.1),
]

# Extract years and uncertainties
years = [entry[2] for entry in predictions]
uncertainties = [entry[3] for entry in predictions]

# Calculate weights as inverse variance (1 / uncertainty²)
weights = [1 / (u ** 2) for u in uncertainties]
print(weights)
# Weighted mean calculation
weighted_mean = np.average(years, weights=weights)

# Weighted standard deviation (standard error of the weighted mean)
weighted_std = np.sqrt(1 / sum(weights))

# 95% confidence interval using normal distribution
z = norm.ppf(0.975)  # z-score for 95% confidence ≈ 1.96
ci_lower = weighted_mean - z * weighted_std
ci_upper = weighted_mean + z * weighted_std

# ---- Output with explanation ----
print("\n📊 Results from Earthquake Prediction Table:\n")
print(f"Weighted Mean Predicted Year: {weighted_mean:.2f}")
print(f"Weighted Standard Deviation: {weighted_std:.2f} years")
print(f"95% Confidence Interval: {ci_lower:.2f} → {ci_upper:.2f}\n")

print("🧮 How these results were computed:")
print("- Each prediction has an associated uncertainty, which we treat as the standard deviation (σ).")
print("- We compute weights as the inverse of the variance (1 / σ²) to give more reliable predictions higher influence.")
print("- The weighted mean is calculated as: ")
print("    weighted_mean = sum(year_i * weight_i) / sum(weight_i)")
print("- The weighted standard deviation (standard error of the mean) is:")
print("    weighted_std = sqrt(1 / sum(weight_i))")
print("- The 95% confidence interval is computed assuming a normal distribution with z-score ~1.96:")
print("    CI_lower = weighted_mean - 1.96 * weighted_std")
print("    CI_upper = weighted_mean + 1.96 * weighted_std")
print("\nThis approach combines all predictions considering their uncertainties to give a statistically robust overall estimate.")

import numpy as np

# Station predictions with uncertainty (standard deviation)
data = [
    ("ARAU", "Malaysia", 2259, 11.15),
    ("KUAL", "Malaysia", 2303, 45.25),
    ("USMP", "Malaysia", 2228, 8.45),
    ("BEHR", "Malaysia", 2270, 46.50),
    ("PHUK", "Thailand", 2241, 10.05),
]

# Extract years and std deviations
years = np.array([row[2] for row in data])
stds = np.array([row[3] for row in data])
variances = stds**2
weights = 1 / variances

# Weighted average and standard error
avg = np.sum(weights * years) / np.sum(weights)
stderr = np.sqrt(1 / np.sum(weights))

# 95% confidence interval (z ≈ 1.96)
ci = 1.96 * stderr
lower, upper = avg - ci, avg + ci

print(f"Weighted prediction: {avg:2f} ± {ci:2f} (95% CI)")
print(f"Range: {lower:.1f} – {upper:.1f}")

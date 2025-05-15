import csv

# Example input arrays
x = [1, 2, 3]
y = [41, 2, 2]

# Make sure both arrays have the same length
if len(x) != len(y):
    raise ValueError("Both arrays must be of the same length.")

# Combine arrays into a list of (x, y) pairs
points = list(zip(x, y))

# Write to a CSV file
with open("points.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y"])  # Header
    writer.writerows(points)

print("CSV file 'points.csv' has been created successfully.")
